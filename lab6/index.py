from train_utils import train
import argparse
from utils import write_to_file, init, find_best_num_workers
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--question', type=str, default='c2')
    args = parser.parse_args()
    question = args.question
    output_file = args.output_file
    print(f'Running question {question}')
    if question == 'q1':
        d = {question: []}
        batch_size = 32
        epochs = 2
        print(f'{batch_size=}')
        while True:
            try:
                a = init(args, batch_size_arg=batch_size)
                print(f"{a['device']=}")
                _, _, warmup_times = train(a['train_loader'], a['device'], a['optimizer'], a['model'], a['criterion'], 1)
                loss, acc, running_times = train(a['train_loader'], a['device'], a['optimizer'], a['model'], a['criterion'], epochs - 1)
                output = {'Top Accuracy': acc, 'Losses': loss, 'Warmup_times': warmup_times, 'Running_times': running_times, 'batch_size': batch_size}
                print(output)
                d[question].append(output)
                batch_size *= 4
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    break
                else:
                    raise e
        write_to_file(d, output_file)
        print(d)
    elif question == 'c3':
        num_workers = 0
        last_running_time = 10000
        d = {question: []}
        while(True):
            a = init(args)
            loss, acc, running_times = train(a['train_loader'],a['device'],a['optimizer'],a['model'],a['criterion'],a['epochs'])
            output = {'Top Accuracy': acc, 'Losses': loss, 'Running_times': running_times, 'num_workers': num_workers}
            d[question].append(output)
            if(running_times[0]["Data loading time"] > last_running_time):
                break
            last_running_time = running_times[0]["Data loading time"]
            num_workers += 4
        write_to_file(d,output_file)
        print(d)
    elif question == 'c4':
        d = {question: []}
        one_worker = 1
        num_worker = find_best_num_workers(output_file)
        #one worker train
        a = init(args,workers=one_worker)
        loss, acc, running_times = train(a['train_loader'],a['device'],a['optimizer'],a['model'],a['criterion'],a['epochs'])
        output = {'Top Accuracy': acc, 'Losses': loss, 'Running_times': running_times, 'num_workers': one_worker}
        d[question].append(output)
        #best worker train
        a = init(args,workers=num_worker)
        loss, acc, running_times = train(a['train_loader'],a['device'],a['optimizer'],a['model'],a['criterion'],a['epochs'])
        output = {'Top Accuracy': acc, 'Losses': loss, 'Running_times': running_times, 'num_workers': num_worker}
        d[question].append(output)
        write_to_file(d, output_file)
        print(d)
    elif question == 'c5':
        d = {question: []}
        a = init(args, device_arg='gpu')
        loss, acc, running_times = train(a['train_loader'],a['device'],a['optimizer'],a['model'],a['criterion'],a['epochs'])
        output = {'Top Accuracy': acc, 'Losses': loss, 'Running_times': running_times, 'device':a['device']}
        d[question].append(output)
        a = init(args, device_arg='cpu')
        loss, acc, running_times = train(a['train_loader'],a['device'],a['optimizer'],a['model'],a['criterion'],a['epochs'])
        output = {'Top Accuracy': acc, 'Losses': loss, 'Running_times': running_times, 'device':a['device']}
        d[question].append(output)
        write_to_file(d, output_file)
        print(d)
    elif question == 'c6':
        optimizers = ['sgd','sgdN','adam','adadelta','adagrad']
        d = {question: []}
        for opt in optimizers:
            a = init(args, optimizer_arg=opt)
            loss, acc, running_times = train(a['train_loader'],a['device'],a['optimizer'],a['model'],a['criterion'],a['epochs'])
            output = {'Top Accuracy': acc, 'Losses': loss, 'Running_times': running_times, 'device':a['device']}
            d[question].append(output)
        write_to_file(d, output_file)
        print(d)
    # elif case == 'c7':
    #     return 'This is case 7'
    # else:
    #     return "Invalid Case"
    

if __name__ == '__main__':
    main()
