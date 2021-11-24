
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import re
    import os
    import numpy as np

    plt.rcParams['figure.dpi'] = 300

    evaluate_dir = '/workspace/rwang/sugar/works/nas/exps/exp2/logs'
    evaluate_log = ['debug_supernet.train.log', 'debug_kernel.train.log', 'debug_depth.train.log', 
                    'debug_width1.train.log', 'debug_width2.train.log', 'debug_width2_speedup.train.log']

    legends = [log.replace('.train.log', '').replace('debug_', '') for log in evaluate_log]
    batchs = []
    inits = []
    forwards = []
    backwards = []
    updates = []
    totals = []

    for log in evaluate_log:

        with open(os.path.join(evaluate_dir, log), 'r') as f:
            lines = f.readlines()

        lines = [line.replace('\n', '') for line in lines if 'batch' in line]

        epochs = [int(re.search('(\d){1,3}it ', line).group().replace('it ', '')) for line in lines]
        indices = np.argsort(epochs)

        batch = [float(re.search('batch=(\d){1,2}.(\d){1,6}', lines[i]).group().replace('batch=', '')) 
                for i in indices]
        init = [float(re.search('init=(\d){1,2}.(\d){1,6}', lines[i]).group().replace('init=', '')) 
                for i in indices]
        forward = [float(re.search('forward=(\d){1,2}.(\d){1,6}', lines[i]).group().replace('forward=', '')) 
                for i in indices]
        backward = [float(re.search('backward=(\d){1,2}(\d|\.)*', lines[i]).group().replace('backward=', ''))
                    for i in indices]
        update = [float(re.search('update=(\d){1,2}.(\d){1,6}', lines[i]).group().replace('update=', ''))
                for i in indices]
        total = [float(re.search('total=(\d){1,2}(\d|\.)*', lines[i]).group().replace('total=', '')) 
                for i in indices]

        batchs.append(batch)
        inits.append(init)
        forwards.append(forward)
        backwards.append(backward)
        updates.append(update)
        totals.append(total)

    plt.figure(figsize=(12, 9))

    data = totals

    plt.subplot(321)
    plt.plot(epochs, np.array(data).T, '.-')
    plt.axhline(np.mean(data), epochs[0], epochs[-1])
    plt.legend(legends, ncol=2)
    plt.xlabel('Iterations')
    plt.ylabel('Time (second)')
    plt.xlim([min(epochs), max(epochs)])
    plt.ylim([0, 3])
    plt.title('Total Time Cost cross tasks')

    plt.subplot(322)
    plt.boxplot(np.array(data).T, labels=[label + '\n' + '%.2f s' % np.mean(data[i]) for i, label in enumerate(legends)])
    plt.axhline(np.mean(data), epochs[0], epochs[-1])
    plt.ylabel('Time (second)')
    plt.ylim([0, 3])

    data = forwards

    plt.subplot(323)
    plt.plot(epochs, np.array(data).T, '.-')
    plt.axhline(np.mean(data), epochs[0], epochs[-1], color='black')
    plt.legend(legends, ncol=2)
    plt.xlabel('Iterations')
    plt.ylabel('Time (second)')
    plt.ylim([0, 1.5])
    plt.xlim([min(epochs), max(epochs)])
    plt.title('Forward Time Cost cross tasks')

    plt.subplot(324)
    plt.boxplot(np.array(data).T, labels=[label + '\n' + '%.2f s' % np.mean(data[i]) for i, label in enumerate(legends)])
    plt.axhline(np.mean(data), epochs[0], epochs[-1], color='black')
    plt.ylabel('Time (second)')
    plt.ylim([0, 1.5])

    data = backwards

    plt.subplot(325)
    plt.plot(epochs, np.array(data).T, '.-')
    plt.axhline(np.mean(data), epochs[0], epochs[-1])
    plt.legend(legends, ncol=2)
    plt.xlabel('Iterations')
    plt.ylabel('Time (second)')
    plt.xlim([min(epochs), max(epochs)])
    plt.ylim([0, 3])
    plt.title('Backward Time Cost cross tasks')

    plt.subplot(326)
    plt.boxplot(np.array(data).T, labels=[label + '\n' + '%.2f s' % np.mean(data[i]) for i, label in enumerate(legends)])
    plt.axhline(np.mean(data), epochs[0], epochs[-1])
    plt.ylabel('Time (second)')
    plt.ylim([0, 3])

    plt.suptitle('The Speedup Solution achieves')
    plt.tight_layout()
    plt.savefig('/workspace/rwang/sugar/works/nas/debug_result/demo.png', dpi=300)
    plt.show()

    print(1)
