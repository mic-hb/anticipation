from argparse import ArgumentParser
from pathlib import Path

from anticipation.convert import events_to_midi
from anticipation.convert import lm_to_midi

if __name__ == '__main__':
    parser = ArgumentParser(description='auditory check for a tokenized dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    parser.add_argument('range', type=int, default=1,
        help='range of items to examine')
    parser.add_argument('vocab', type=str, default='triplet-midi', help='vocab type: triplet-midi or local-midi')

    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue

            if i == args.index+args.range:
                break
            
            tokens = [int(token) for token in line.split()]

            if args.vocab == 'triplet-midi':
                from anticipation.vocabs.tripletmidi import vocab
                print(f"sonify token length: {len(tokens)}")
                tokens = [tok for tok in tokens if tok < vocab['special_offset']]
                print(f"after removing special offset: {len(tokens)}")
                assert(len(tokens) % 3 == 0)
                mid = events_to_midi(tokens, vocab)
            else: # vocab = local-midi
                from anticipation.vocabs.localmidi import vocab
                # tokens = [tok for tok in tokens if tok < vocab['special_offset']]
                mid = lm_to_midi(tokens, vocab)
            mid.save(f'output/{Path(args.filename).stem}{i}.mid')
