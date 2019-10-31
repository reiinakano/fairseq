#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from sympy.parsing.sympy_parser import parse_expr


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=True)

    # Initialize generator
    num_sentences = 0
    has_target = True
    model = models[0]
    generator, scorer = None, None
    total, correct = 0, 0
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            model.eval()
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in sample['net_input'].items()
                if k != 'prev_output_tokens'
            }
            src_tokens = encoder_input['src_tokens']
            src_lengths = (src_tokens.ne(tgt_dict.eos()) & src_tokens.ne(tgt_dict.pad())).long().sum(dim=1)
            input_size = src_tokens.size()
            # batch dimension goes first followed by source lengths
            bsz = input_size[0]
            src_len = input_size[1]

            if args.verbose:
                print('SRCLENGTHS', src_lengths, 'BSZ', bsz, 'SRCLEN', src_len)
                print('SRCTOKENSSHAPE', src_tokens.shape)

            for sample_iter in range(bsz):
                print('\n')
                with torch.no_grad():
                    single_src_lengths = encoder_input['src_lengths'][sample_iter:sample_iter+1]
                    single_src_tokens = encoder_input['src_tokens'][sample_iter:sample_iter+1]
                    single_target_tokens = sample['net_input']['prev_output_tokens'][sample_iter:sample_iter+1]

                    if args.verbose:
                        print('SINGLE SRC TOKENS', single_src_tokens, 'SINGLE SRC LENGTHS', single_src_lengths)
                        print('SINGLE TGT TOKENS', single_target_tokens)

                    question_str = ''
                    for i in range(len(single_src_tokens[0])):
                        question_str += src_dict[single_src_tokens[0][i]]
                    tgt_str = ''
                    for i in range(len(single_target_tokens[0])):
                        tgt_str += tgt_dict[single_target_tokens[0][i]]

                    encoder_out = model.encoder.forward(single_src_tokens, single_src_lengths)
                    if args.verbose:
                        print(encoder_out)
                        print('encoder_out shape', encoder_out['encoder_out'].shape,
                              'encoder_embedding shape', encoder_out['encoder_embedding'].shape)

                    prev_output_tokens_list = [tgt_dict.eos()]
                    token_idx = 0
                    symbolic_calculator = SymbolicCalculator()
                    while prev_output_tokens_list[-1] != tgt_dict.eos() or len(prev_output_tokens_list) == 1:
                        prev_output_tokens = torch.LongTensor([prev_output_tokens_list]).to(encoder_out['encoder_out'].device)
                        decoder_out = model.decoder.forward(prev_output_tokens, encoder_out)
                        print(decoder_out[1]['attn'], decoder_out[1]['attn'].shape)
                        raise
                        decoder_out = decoder_out[0][0][token_idx]
                        #print('decoder output shape', decoder_out.shape)
                        top_indices = decoder_out.argsort(descending=True)
                        if args.verbose:
                            print('\n')
                            top_indices_str = ['top 5 values']
                            for i in range(5):
                                top_indices_str.append(' {:.2f} : "{}" '.format(decoder_out[top_indices[i]].item(), tgt_dict[top_indices[i]]))
                            print('|'.join(top_indices_str))
                        prev_output_tokens_list.append(top_indices[0].item())
                        calc_response = symbolic_calculator.press(tgt_dict[top_indices[0].item()])
                        if calc_response != '':  # If calculator responds (to an = sign)
                            if calc_response == '<err>':
                                answer_so_far_str = 'calculator error'
                                break
                            else:
                                for char in calc_response:
                                    prev_output_tokens_list.append(tgt_dict.index(char))
                                prev_output_tokens_list.append(tgt_dict.index('@'))
                                token_idx += len(calc_response) + 1

                        answer_so_far_str = ''
                        for ind in prev_output_tokens_list:
                            answer_so_far_str += tgt_dict[ind]
                        token_idx += 1

                    def trim_padding_and_eos(x: str):
                        x = x.replace('<pad>', '')
                        x = x.replace('</s>', '')
                        return x

                    question_str_trimmed = trim_padding_and_eos(question_str)
                    tgt_str_trimmed = trim_padding_and_eos(tgt_str)
                    answer_so_far_str_trimmed = trim_padding_and_eos(answer_so_far_str)

                    print('[QUESTION]', question_str_trimmed)
                    print('[TARGET ANSWER]', tgt_str_trimmed)
                    print('[PREDICTION]', answer_so_far_str_trimmed)

                    actual_answer = tgt_str_trimmed.split('@')[-1]
                    actual_prediction = answer_so_far_str_trimmed.split('@')[-1]
                    if actual_answer == actual_prediction:
                        print('Prediction correct')
                        correct += 1
                    else:
                        print('Prediction incorrect')
                    total += 1
                    print('[AVERAGE SCORE SO FAR]: {}/{} = {:.3f}'.format(correct, total, float(correct)/total))



    #if has_target:
    #    print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))


class SymbolicCalculator():
    def __init__(self, syntax_error_symbol='<err>'):
        self.current_equation = ''
        self.syntax_error_symbol = syntax_error_symbol

    def press(self, x: str):
        if x == '=':
            eq = self.current_equation
            solution = self.solve_current_equation()
            print('[Symbolic Calculator] calculator responded {}={}'.format(eq, solution))
            return solution
        elif x == '@':
            self.current_equation = ''
            return ''
        else:
            self.current_equation += x
            return ''

    def solve_current_equation(self):
        try:
            solution = parse_expr(self.current_equation)
        except SyntaxError:
            return '<err>'
        self.current_equation = ''
        return str(solution)


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
