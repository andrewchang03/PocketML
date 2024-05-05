import asyncio
import time
import logging
import multiprocessing
import requests
import pickle

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

state = {
    'train_process': None,  # process for training loop
    'train_function': None,  # target training function
    'train_config': {
        'batch_size': 64,
        'epochs': 14,
        'lr': 1.0,
        'log_interval': 10
    },
    'train_status': True,  # True if running
    'job_name': '',  # name of the current job
    'using_wandb': False,  # is training using wandb?
    'wandb_link': '',  # link to wandb dashboard
    'job_id': 0,  # current job id
    'current_step': 0  # current step (current batch * epoch)
}

def init(train):
    state['train_function'] = train

    event_loop = asyncio.get_event_loop()  # background event loop for api requests

    try:
        start_training()  # start training by default when you first start running
        # asyncio.ensure_future(stop_training())
        asyncio.ensure_future(constant_update())
        event_loop.run_forever()
    finally:
        logging.info("Closing Loop")
        event_loop.close()

def set_job_info(job_name, using_wandb, wandb_link):
    state['job_name'] = job_name
    state['using_wandb'] = using_wandb
    state['wandb_link'] = wandb_link

def set_current_step(step):
    state['current_step'] = step

def post_new_job():
    payload = {
        "name": state['job_name'], 
        "wandb": state['using_wandb'],
        "wandb_link": state['wandb_link'],
        "start_time": int(time.time()),
        "config": state['train_config'],
        "cluster_name": "local",
        "total_steps": 0,
    }

    res = requests.post(
        'http://10.48.85.83:8000/api/v1/jobs/new_job', 
        json=payload, 
        headers={'email': 'a@a.com', 'password': 'abcdef'}
    )

    state['job_id'] = res.json()['detail']
    f = open('job_id', 'wb')
    pickle.dump(state['job_id'], f)

def start_training():
    # puts the training function into a process
    logging.info("Creating training process")

    state['train_process'] = multiprocessing.Process(
        target=state['train_function'], 
        args=(state['train_config'],),
        daemon=True
    )

    # start training process
    logging.info("Starting training")
    state['train_process'].start()

async def stop_training():
    while True:
        await asyncio.sleep(10)
        if state['train_process'] and state['train_status']:
            state['train_process'].terminate()

async def constant_update():
    while True:
        await asyncio.sleep(1)

        # constantly update current run status to api server
        payload = {
            "step": state['current_step'],
            "status": "running",
            "update_time": int(time.time())
        }

        f = open('job_id', 'rb')
        job_id = pickle.load(f)

        res = requests.post(
            'http://10.48.85.83:8000/api/v1/jobs/' + str(job_id) +'/update', 
            json=payload, 
            headers={'email': 'a@a.com', 'password': 'abcdef'}
        )

        logging.info('Updated to Server: ' + str(res))

        print(res)
        action = res.json()['action']

        if action == "start":
            if not state['train_status']:
                start_training()
            state['train_status'] = True
        elif action == "stop":
            if state['train_status']:
                state['train_process'].terminate()
            state['train_status'] = False

async def get_change():
    while True:
        await asyncio.sleep(1)
        f = open('job_id', 'rb')
        job_id = pickle.load(f)
        res = requests.get('http://10.48.85.83:8000/api/v1/jobs/' + str(job_id) + '/get_change')
        res = res.json()
        if res['is_changed']:
            train_config = res['config']
            state['train_process'].terminate()
            start_training()