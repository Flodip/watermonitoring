import queue
import threading
import time

q = queue.Queue()


class ProducerThread(threading.Thread):
    def __init__(self, data, time_between_samples=1, target=None, name=None):
        super(ProducerThread, self).__init__()
        self.target = target
        self.name = name
        self.data = data
        self.time_between_samples = time_between_samples

    def run(self):
        while True:
            t = threading.Thread(target=time.sleep, args=(self.time_between_samples-0.001,), daemon=True)
            t.start()
            q.put(self.data())
            t.join()


class ConsumerThread(threading.Thread):
    def __init__(self, process_data, wait_between_two_pulls=3, pool_size=10, target=None, name=None):
        super(ConsumerThread, self).__init__()
        self.target = target
        self.name = name
        self.pool_size = pool_size
        self.process_data = process_data
        self.wait_between_two_pulls = wait_between_two_pulls

    def run(self):
        while True:
            time.sleep(self.wait_between_two_pulls)
            print("queue size: " + str(q.qsize()))
            if q.qsize() >= self.pool_size:
                print("#threads: " + str(threading.active_count()), flush=True)
                print("    #samples to add: " + str(q.qsize()), flush=True)
                self.process_data(q, self.pool_size)


if __name__ == '__main__':
    def prod():
        print('call prod')
        return 5

    def cons(q, pool_size):
        print('call cons')
        print(q.get())

    p = ProducerThread(prod, name='producer')
    c = ConsumerThread(cons, name='consumer', pool_size=20)

    p.start()
    c.start()