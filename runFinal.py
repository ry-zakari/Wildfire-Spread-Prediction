import argparse
import os
import subprocess
filename = '/trainModel.py'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master', default='mars')
    parser.add_argument('-p', '--port', default='7669')
    parser.add_argument('-n', '--nodes', default="earth",
                        help='list of nodes, no master')
    parser.add_argument('--epochs', default=1, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    print(f'Master: {args.master}')
    nodes = args.nodes.split(' ')
    print(f'Nodes: ', end='')
    for node in nodes:
        print(f'{node} ', end='')
    print()
    num_nodes = len(nodes) + 1
    print(f'Starting Master: {args.master} on Port: {args.port}')
    subprocess.Popen(f'ssh {args.master} "python3 {filename} -m {args.master} -p {args.port} -n {num_nodes} -g 1 -nr 0 --epochs {args.epochs}"',shell=True)
    for i, node in enumerate(nodes):
        print(f'Starting Node: {node}')
        subprocess.Popen(
            f'ssh {node} "python3 {filename} -m {args.master} -p {args.port} -n {num_nodes} -g 1 -nr {i+1} --epochs {args.epochs}"', shell=True)



if __name__ == '__main__':
    main()