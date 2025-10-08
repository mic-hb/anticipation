import os, math, traceback
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from tqdm import tqdm
from pathlib import Path
from functools import partial

import numpy as np

from anticipation import ops
from anticipation import tokenize

def prepare_local_midi(midifile, tripletmidivocab, localmidivocab, task, transcript):
    with open(midifile, 'r') as f:
        events, truncations, status = tokenize.maybe_tokenize([int(token) for token in f.read().split()])
        # print(f"max of events: {max(events)}")
    if status > 0:
        return events, [], [], status
    
    if task == 'autoregress':
        z = [localmidivocab['control_end']] # BOS tag, could be more in the future
        # no "if transcript"
    else: # anticipation, implementing later
        pass
    
    time_res = localmidivocab["config"]["midi_quantization"]
    recent_tick = 0
    tokens = []

    # separators are in groups of three here, strip them out
    # fix the note durations by subtracting offset and then adding the note offset
    for time, dur, note in zip(events[0::3], events[1::3], events[2::3]):
        if time != tripletmidivocab['separator'] and dur != tripletmidivocab['separator'] and note != tripletmidivocab['separator']: # separator in convert.py is pulled from vocab.py but it has same value as tripletmidivocab's separator (i.e. 55025) --> skip any of the separators
            while time >= round(recent_tick * time_res):
                tokens.append(localmidivocab['tick']) # insert tick
                recent_tick += 1
            relativize = round((recent_tick - 1) * time_res) if recent_tick > 0 else 0
            tokens.append(time - relativize)
            # print(f"time: {time - relativize}")
            tokens.append(dur - tripletmidivocab['duration_offset'] + localmidivocab['duration_offset'])
            # print(f"dur: {dur}")
            tokens.append(note - tripletmidivocab['note_offset'] + localmidivocab['note_offset']) 
            # print(f"note: {note}")
    
    separator = [localmidivocab['separator']]
    return tokens, z, separator, 0

def prepare_triplet_midi(midifile, vocab, task, transcript):
    with open(midifile, 'r') as f:
        events, truncations, status = tokenize.maybe_tokenize([int(token) for token in f.read().split()])

    if status > 0:
        return events, [], [], status

    # record the original end time before extracting control tokens
    end_time = ops.max_time(events, seconds=False)

    if  task == 'autoregress':
        z = [vocab['task']['autoregress']]
        if transcript:
            z = [vocab['task']['autoregress_transcript']]

        controls = []
    else:
        z = [vocab['task']['anticipate']]
        if transcript:
            z = [vocab['task']['anticipate_transcript']]

        if task == 'instrument':
            instruments = list(ops.get_instruments(events).keys())
            if len(instruments) < 2:
                # need at least two instruments for instrument anticipation
                return events, 4 # status 4 == too few instruments

            u = 1+np.random.randint(len(instruments)-1)
            subset = np.random.choice(instruments, u, replace=False)
            events, controls = tokenize.extract_instruments(events, subset)
        elif task == 'span':
            events, controls = tokenize.extract_spans(events, .05)
        elif task == 'random':
            events, controls = tokenize.extract_random(events, 10)

    # add rest tokens to events after extracting control tokens
    # (see Section 3.2 of the paper for why we do this)
    events = ops.pad(events, end_time)

    # interleave the events and anticipated controls
    tokens, controls = ops.anticipate(events, controls)
    assert len(controls) == 0 # should have consumed all controls (because of padding)

    separator = [vocab['separator'] for _ in range(3)]
    return tokens, z, separator, 0


def pack_tokens(sequences, output, idx, prepare, factor, config, seqlen, vocab):
    vocab_size = config['size']
    max_arrival = config['max_time']
    log = output + '.log'

    seqcount = 0
    stats = 5*[0] # (short, long, too many instruments, too few instruments, inexpressible)
    with open(output, 'w') as outfile:
        concatenated_tokens = []
        z_tokens = []
        for sequence in tqdm(sequences, desc=f'#{idx}', position=idx+1, leave=True):
            with open(log, 'a') as f:
                f.write(sequence + '\n')

            for _ in range(factor):
                tokens, z, separator, status = prepare(sequence)
                #print(f"max of tokens: {max(tokens)}")

                if status > 0:
                    stats[status-1] += 1
                    break

                # write out full contexts to file
                # separator is None for local-midi
                concatenated_tokens.extend(separator + z + tokens) # this is entirely new sequence being concatenated to list, so new z
                z_tokens.extend([z] * len(separator + z + tokens))

                z = z_tokens[0] # need to redefine z to be the correct z, not just most recent
                while len(concatenated_tokens) >= seqlen-len(z):
                    seq = concatenated_tokens[0:seqlen-len(z)]
                    z_seq = z_tokens[0:seqlen-len(z)]
                    z = z_seq[0]
                    
                    concatenated_tokens = concatenated_tokens[len(seq):]
                    z_tokens = z_tokens[len(seq):]

                    seq = z + seq
                    
                    assert max(seq) < vocab_size

                    outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                    seqcount += 1

    return (seqcount, *stats)


def init_worker(lock):
    tqdm.set_lock(lock)
    np.random.seed(os.getpid())

def main(args):
    print('Tokenizing a dataset at:', args.datadir)

    if args.vocab == 'triplet-midi':
        from anticipation.vocabs.tripletmidi import vocab as tripletmidivocab
        prepare = partial(prepare_triplet_midi, vocab=tripletmidivocab, task=args.task, transcript=args.transcript)
        #lambda midifile: prepare_triplet_midi(midifile, vocab, args.task, args.transcript)
    elif args.vocab == 'local-midi':
        from anticipation.vocabs.tripletmidi import vocab as tripletmidivocab
        from anticipation.vocabs.localmidi import vocab as localmidivocab

        # print("triplet vocab keys:", tripletmidivocab.keys())
        # print("local vocab keys:", localmidivocab.keys())

        prepare = partial(prepare_local_midi, tripletmidivocab=tripletmidivocab, localmidivocab=localmidivocab, task=args.task, transcript=args.transcript)
    else:
        raise ValueError(f'Invalid vocabulary type "{args.vocab}"')

    if args.task == 'autoregress':
        assert args.factor == 1, f'Autoregressive preprocessing has no randomness; cannot apply augmentation factor {factor}'

    print('Tokenization parameters:')
    print(f"  vocab = {args.vocab}")
    print(f"  task = {args.task}")
    print(f"  context = {args.context}")
    print(f"  augmentation factor = {args.factor}")
    print(f"  transcript? {args.transcript}")

    files = glob(os.path.join(args.datadir, '**/*.compound.txt'), recursive=True)
    #print(f"files: {files}")

    n = len(files) // args.workers
    shards = [files[i*n:(i+1)*n] for i in range(args.workers)] # dropping a few tracks (< args.workers)
    outfiles = os.path.join(args.outdir, os.path.basename(args.datadir) + '.{t}.shard-{s:03}.txt')
    print('Outputs to:', outfiles)
    outputs = [outfiles.format(t=args.task, s=s) for s in range(len(shards))]
    prepare = args.workers*[prepare]
    context = args.workers*[args.context]
    task = args.workers*[args.task]
    factor = args.workers*[args.factor]
    transcript = args.workers*[args.transcript]
    # config = args.workers*[{**vocab['config'], 'vocab': args.vocab}] # this worked before for triplet-midi
    
    if args.vocab == 'local-midi':
        default_config = localmidivocab['config'].copy()
    else:
        default_config = tripletmidivocab['config'].copy()

    default_config['vocab'] = args.vocab

    # Set a default if 'max_time' is not defined
    if 'max_time' not in default_config:
        default_config['max_time'] = float('inf')  # or a large constant

    config = args.workers * [default_config]

    chosen_vocab = localmidivocab if args.vocab == 'local-midi' else tripletmidivocab

    print('Processing...')
    if args.debug:
        results = pack_tokens(shards[0], outputs[0], 0, prepare, args.factor, config[0], args.context, chosen_vocab)
        results = [results]
    else:
        with Pool(processes=args.workers, initargs=(RLock(),), initializer=init_worker) as pool:
            results = pool.starmap(pack_tokens, zip(shards, outputs, range(args.workers), prepare, factor, config, context, args.workers*[chosen_vocab]))

    seqcount, too_short, too_long, many_instr, few_instr, discarded_seqs = (sum(x) for x in zip(*results))

    print('Tokenization complete.')
    print(f'  => Processed {seqcount} training sequences')
    print(f'  => Discarded {too_short+too_long+many_instr+few_instr} event sequences')
    print(f'      - {too_short} too short')
    print(f'      - {too_long} too long')
    print(f'      - {many_instr} too many instruments')
    print(f'      - {few_instr} too few instruments')
    print(f'  => Discarded {discarded_seqs} training sequences')


if __name__ == '__main__':
    parser = ArgumentParser(description='tokenizes a dataset')
    parser.add_argument('datadir', help='directory containing the dataset to tokenize')
    parser.add_argument('outdir', help='location to store the tokenized datafile')
    parser.add_argument('task', help='task for which we are preparing sequences')
    parser.add_argument('context', type=int, help='context length for packing training sequences')
    parser.add_argument('-v', '--vocab', default='triplet-midi', help='name of vocabulary to use for tokenization')
    parser.add_argument('-f', '--factor', type=int, default=1, help='augmentation factor')
    parser.add_argument('-t', '--transcript', action='store_true', help='transcribed midi file')
    parser.add_argument('--workers', type=int, default=16, help='number of workers/shards')
    parser.add_argument('--debug', action='store_true', help='debugging (single shard; non-parallel)')

    main(parser.parse_args())
