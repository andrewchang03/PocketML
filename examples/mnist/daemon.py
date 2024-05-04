# import requests
import asyncio
import time
import logging
import threading
import multiprocessing

config = {
    'batch_size': 64,
    'epochs': 14,
    'lr': 1.0,
    'log_interval': 10
}

def init(train):
    # basic logging configurations
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # puts the training function into a process
    logging.info("Creating training process")
    train_process = multiprocessing.Process(target=train, args=(config,))

    # start training process
    logging.info("Starting training")
    train_process.start()

    async def stop_training(train_process):
        while True:
            await asyncio.sleep(5)
            train_process.terminate()

    # event loop for api requests
    event_loop = asyncio.get_event_loop()

    try:
        asyncio.ensure_future(stop_training(train_process))
        event_loop.run_forever()
    finally:
        print("Closing Loop")
        event_loop.close()