import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mylist', nargs='+', help='List of values')
args = parser.parse_args()
print(args.mylist)