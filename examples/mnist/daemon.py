import asyncio
import time
import logging
import multiprocessing
import requests

from enum import Enum

class Status(Enum):
    RUNNING = 0
    STOPPED = 1
    PAUSED = 2

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

def init(train):
    global train_process, train_function, train_config, train_status
    global job_name, using_wandb, wandb_link
    global job_id
    global current_step

    train_process = None  # process for training loop
    train_function = train  # target training function
    train_config = {
        'batch_size': 64,
        'epochs': 14,
        'lr': 1.0,
        'log_interval': 10
    }
    train_status = Status.RUNNING

    job_name = ''  # name of the current job
    using_wandb = False  # is training using wandb?
    wandb_link = ''  # link to wandb dashboard

    job_id = 0  # current job id
    current_step = 0  # current step (current batch * epoch)

    event_loop = asyncio.get_event_loop()  # background event loop for api requests

    try:
        start_training()  # start training by default when you first start running
        asyncio.ensure_future(stop_training())
        asyncio.ensure_future(constant_update())
        event_loop.run_forever()
    finally:
        logging.info("Closing Loop")
        event_loop.close()

def get_job_info(name, wandb, wandb_url):
    global job_name, using_wandb, wandb_link
    job_name = name
    using_wandb = wandb
    wandb_link = wandb_url

def get_current_step(step):
    global current_step
    current_step = step

def start_training():
    global train_process, train_function, train_config, train_status
    global job_id

    # puts the training function into a process
    logging.info("Creating training process")
    train_process = multiprocessing.Process(
        target=train_function, 
        args=(train_config,),
        daemon=True
    )

    # post new job to api server
    payload = {
        "name": job_name, 
        "wandb": using_wandb,
        "wandb_link": wandb_link,
        "start_time": int(time.time()),
        "config": train_config,
        "cluster_name": "local",
        "total_steps": 0,
    }

    res = requests.post(
        'http://10.48.85.83:8000/api/v1/jobs/new_job', 
        json=payload, 
        headers={'email': 'a@a.com', 'password': 'abcdef'}
    )

    job_id = res.json()['detail']

    # start training process
    logging.info("Starting training")
    train_process.start()

async def stop_training():
    while True:
        await asyncio.sleep(10)
        if train_process and train_status == Status.STOPPED:
            train_process.terminate()

async def constant_update():
    global train_process, train_status

    while True:
        await asyncio.sleep(1)

        # constantly update current run status to api server
        payload = {
            "step": current_step,
            "status": "running",
            "update_time": int(time.time())
        }

        res = requests.post(
            'http://10.48.85.83:8000/api/v1/jobs/' + str(job_id) +'/update', 
            json=payload, 
            headers={'email': 'a@a.com', 'password': 'abcdef'}
        )

        logging.info('Updated to Server: ' + str(res))

        action = res.json()['action']

        if action == "start":
            if train_status == Status.STOPPED:
                start_training()
            train_status = Status.RUNNING
        elif action == "stop":
            if train_status == Status.RUNNING:
                train_process.terminate()
            train_status = Status.STOPPED
        else:
            train_status = Status.PAUSED

async def get_change():
    global train_process, train_config

    while True:
        await asyncio.sleep(1)
        res = requests.get('http://10.48.85.83:8000/api/v1/jobs/' + str(job_id) + '/get_change')
        res = res.json()
        if res['is_changed']:
            train_config = res['config']
            train_process.terminate()
            start_training()