import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, key=None, emb_name=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            
            output = []
            print(f'All runs:')
            print(key)
            output.append('All runs:')
            output.append(key)
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            output.append(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            output.append(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            output.append(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            output.append(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            with open(f'hl_gnn_planetoid/metrics_and_weights/results_{emb_name}_ft.txt', 'a') as f:
                for line in output:
                    print(line)
                    f.write(line + '\n')
                f.write('\n')

    def print_statistics_others(self, run=None, key=None, emb_name=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result.argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'   Final Test: {result[argmax]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                test = r[r.argmax()].item()
                best_results.append((test))

            best_result = torch.tensor(best_results)
            
            output = []
            print(f'All runs:')
            print(key)
            output.append('All runs:')
            output.append(key)
            output.append(f'Highest Train: nan ± nan')
            output.append(f'Highest Valid: nan ± nan')
            output.append(f'  Final Train: nan ± nan')
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            output.append(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            with open(f'hl_gnn_planetoid/metrics_and_weights/results_{emb_name}_ft.txt', 'a') as f:
                for line in output:
                    print(line)
                    f.write(line + '\n')
                f.write('\n')