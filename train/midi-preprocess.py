import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob

from tqdm import tqdm

from anticipation.convert import midi_to_compound
from anticipation.config import PREPROC_WORKERS


def convert_midi(filename, debug=False):
    try:
        tokens = midi_to_compound(filename, debug=debug)
    except Exception:
        if debug:
            print('Failed to process: ', filename)
            print(traceback.format_exc())

        return 1

    with open(f"{filename}.compound.txt", 'w') as f:
        f.write(' '.join(str(tok) for tok in tokens))

    return 0


def main(args):
    filenames = glob(args.dir + '/**/*.mid', recursive=True) \
            + glob(args.dir + '/**/*.midi', recursive=True)

    workers = getattr(args, 'workers', PREPROC_WORKERS) or PREPROC_WORKERS

    print(f'Preprocessing {len(filenames)} files with {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(convert_midi, filenames), desc='Preprocess', total=len(filenames)))

    discards = round(100*sum(results)/float(len(filenames)),2)
    print(f'Successfully processed {len(filenames) - sum(results)} files (discarded {discards}%)')

if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a MIDI dataset')
    parser.add_argument('dir', help='directory containing .mid files for training')
    parser.add_argument(
        '--workers',
        type=int,
        default=PREPROC_WORKERS,
        help='number of parallel workers',
    )
    main(parser.parse_args())
