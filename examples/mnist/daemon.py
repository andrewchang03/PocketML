# import requests
import asyncio
import time
import logging
import threading

config = {
    'batch_size': 64,
    'epochs': 14,
    'lr': 1.0,
    'log_interval': 10
}

# async def api_response():
#     while True:
#         await signal from api server
#         if run:
#             # start a subprocess to execute python file in paths
#             pass
#         elif stop:
#             pass
#         elif resume:
#             pass
#         # also need to think about whether there are new configs

def init(train):
    # requests.post() # post to api server that project is live and running
    # # on 
    # # expected params: PocketML username, cluster name

    # configure logging output
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    logging.info("Creating training thread")
    train_thread = threading.Thread(target=train, args=(config,), daemon=True)

    logging.info("Starting training")
    train_thread.start()

    event_loop = asyncio.get_event_loop()

    try:
        # asyncio.ensure_future(firstWorker())
        # asyncio.ensure_future(secondWorker())
        event_loop.run_forever()
    finally:
        print("Closing Loop")
        event_loop.close()

    # logging.info("Main    : before running thread")
    # x.start()
    # logging.info("Main    : wait for the thread to finish")
    # # x.join()
    # while True:
    #     time.sleep(1)
    #     print("hello")
    # logging.info("Main    : all done")