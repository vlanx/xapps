from fastapi import FastAPI, Request
from kubernetes import client, config, watch
import aiohttp
import asyncio
from os import path

import yaml
import requests
import logging
import time

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

session = requests.Session()
session.verify = False

CURRENT_DEPLOYMENT = "deployment1"

CSMFIP = "10.16.10.74:8000"

api_url_low = "http://"+CSMFIP+"/productOrder/LowQuality/patch"
api_url_high = "http://"+CSMFIP+"/productOrder/HighQuality/patch"

config.load_kube_config(config_file="./kubeconfig.yaml")

def create_deployment(filename):

    with open(path.join(path.dirname(__file__), filename)) as f:
        dep = yaml.safe_load(f)
        k8s_apps_v1 = client.AppsV1Api()
        k8s_core_v1 = client.CoreV1Api()

        resp = k8s_apps_v1.create_namespaced_deployment(
            body=dep, namespace="default")
        if filename == "deployment.yaml":
            logger.info("Created Chart1 Deployment")
        else:
            logger.info("Created Chart2 Deployment")

    deployment_name = dep['metadata']['name']
    namespace = 'default'

    w = watch.Watch()
    for event in w.stream(k8s_core_v1.list_namespaced_pod, namespace=namespace, timeout_seconds=600):
        pod = event['object']
        pod_status = pod.status.phase

        if pod_status == 'Running':
            logger.info(f"Pod is running")
            break
    
    print('Done')
    
        
         
def delete_deployment(filename):
    with open(path.join(path.dirname(__file__), filename)) as f:
        dep = yaml.safe_load(f)
        k8s_apps_v1 = client.AppsV1Api()
        if filename == "deployment.yaml":
            resp = k8s_apps_v1.delete_namespaced_deployment(
                name="chart1-deployment", namespace="default")
            logger.info("Deleted Chart1 Deployment")
        else:
            resp = k8s_apps_v1.delete_namespaced_deployment(
                name="chart2-deployment", namespace="default")
            logger.info("Deleted Chart2 Deployment")
            
async def call_nef(url):
    async with aiohttp.ClientSession() as session:
        nef_start_time = time.time()  
        async with session.patch(url,json={"administrative_state": "UNLOCKED","operational_state": "ENABLED"}) as response:
            resp = await response.json()
            nef_end_time = time.time()
            nef_elapsed_time = nef_end_time - nef_start_time
            if resp["description"]!="Success":
                logger.info(f"Changed Network Slice ({nef_elapsed_time} seconds)")
                
@app.get("/")
async def root():
    
    global CURRENT_DEPLOYMENT 
    
    if CURRENT_DEPLOYMENT == "deployment1":
        dep2_start_time = time.time()
    
        create_deployment("deployment2.yaml")
        dep2_end_time = time.time()
        dep2_elapsed_time = dep2_end_time - dep2_start_time
        logger.info(f"Created Deployment 2 ({dep2_elapsed_time} seconds)")

        await call_nef(api_url_high)
        delete_deployment("deployment.yaml")
        
        CURRENT_DEPLOYMENT = "deployment2"
        
    else:
        dep1_start_time = time.time()
        create_deployment("deployment.yaml")
        dep1_end_time = time.time()
        dep1_elapsed_time = dep1_end_time - dep1_start_time
        logger.info(f"Created Deployment 1 ({dep1_elapsed_time} seconds)")

        await call_nef(api_url_low)
        delete_deployment("deployment2.yaml")
          
        CURRENT_DEPLOYMENT = "deployment1"
            
    return {"message": "Deployment Created"}

@app.post("/log")
async def log(log: dict):
    log_message = log.get('log', '')
    logger.info(log_message)

    return {"received_string": log_message}
