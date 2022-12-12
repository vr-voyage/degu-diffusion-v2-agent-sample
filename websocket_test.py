import asyncio
import base64
import hashlib
import json
import functools
import traceback

from myylibs.jobsmanager import JobQueue, Job, StatusReport
from sdworker import DeguDiffusionWorker

class MyQueue(JobQueue):
    def report_job_started(self, job:Job, report:StatusReport):
        job.external_reference.report(message = "Your job has started !")

    def report_job_done(self, job:Job, report:StatusReport):
        job.external_reference.report(message = "Job finished ! Thanks for using Degu Diffusion !")

    def report_job_progress(self, job:Job, report:StatusReport):
        result = report.result
        if type(result) != dict:
            job.external_reference.report_error(message = "Something went wrong. Yell at a dev !")
            return
        
        if not result.keys() >= {"filepath", "nsfw", "seed"}:
            job.external_reference.report_error(message = "The image generator is not reporting results correctly. Yell at a dev !")
            return

        if result["nsfw"]:
            job.external_reference.report_rerun(cause = "nsfw")
            return

        job.external_reference.report_result(image_data = result["image_data"].read(), prompt = result["actual_prompt"], seed = result["seed"] )

        
    def report_job_failed(self, job:Job, report:StatusReport):
        job.external_reference.report_error("Ow... The whole thing broke... Try again later, maybe !")
    
    def report_job_canceled(self, job:Job, report:StatusReport):
        job.external_reference.report_cancel("Job canceled")

class JobRequest:
    def __init__(this, websocket, internal_id):
        this.websocket = websocket
        this.internal_id = internal_id
    
    def base_response(this, type:str) -> dict:
        response = { 
            "version": "0",
            "internal_id": f"{this.internal_id}",
            "type": f"@degudiffusion_{type}"
        }
        return response

    def message_response(this, message:str, type:str) -> dict:
        response:dict = this.base_response(type)
        response["message"] = message
        return response

    def rerun_response(this, cause:str) -> dict:
        response:dict = this.base_response(type)
        response["cause"] = cause
        return response

    def image_response_for(this, image_data:bytes, prompt:str, seed:str) -> str:
        image_data_base64   = base64.b64encode(image_data)
        image_data_header   = "data:image/png;base64,".encode("ascii")
        image_data_complete = image_data_header + image_data_base64
        image_data_size     = len(image_data_complete)
        image_data_digest   = hashlib.sha256(image_data_complete).hexdigest()
        response            = this.base_response("txt2img_response")

        response["prompt"] = prompt
        response["seed"]   = seed
        response["digest"] = image_data_digest
        response["size"]   = image_data_size
        response["data"]   = image_data_complete.decode('ascii')
        return response

    def websocket_send(this, response:dict):
        asyncio.ensure_future(this.websocket.send(json.dumps(response)))

    def report(this, message:str):
        this.websocket_send(this.message_response(message = message, type = "report_progress"))
    
    def report_error(this, message:str):
        this.websocket_send(this.message_response(message = message, type = "report_error"))
    
    def report_rerun(this, cause:str):
        this.websocket_send(this.rerun_response(cause = cause, type = "rerun"))

    def report_result(this, image_data:bytes, prompt:str, seed:str):
        this.websocket_send(this.image_response_for(image_data = image_data, prompt = prompt, seed = seed))
    

class SDWebsocketWorker():
    def __init__(self):
        config = {
            'COMPACT_RESPONSES': False,
            'DEFAULT_IMAGES_PER_JOB': 8,
            'DEFAULT_PROMPT': 'Degu enjoys its morning coffee by {random_artists}, {random_tags}',
            'HUGGINGFACES_TOKEN': '',
            'IMAGES_HEIGHT': 512,
            'IMAGES_WIDTH': 512,
            'MAX_IMAGES_PER_JOB': 64,
            'MAX_IMAGES_BEFORE_THREAD': 2,
            'MAX_INFERENCES_PER_IMAGE': 120,
            'MAX_GUIDANCE_SCALE_PER_IMAGE': 30,
            'OUTPUT_DIRECTORY': 'images',
            'STABLEDIFFUSION_CACHE_DIR': 'stablediffusion_cache',
            'STABLEDIFFUSION_LOCAL_ONLY': False,
            'STABLEDIFFUSION_MODEL_NAME': 'CompVis/stable-diffusion-v1-4',
            'STABLEDIFFUSION_MODE': 'fp32',
            'SAVE_IMAGES_TO_DISK': True,
            'TORCH_DEVICE': 'cuda'
        }

        self.queue = JobQueue(self.generate_worker, functools.partial(self.generate_worker, config))

if __name__ == '__main__':
    import asyncio
    from websockets import serve

    def generate_worker():
        config = get_config()
        return DeguDiffusionWorker(
            model_name    = 'CompVis/stable-diffusion-v1-4',
            sd_token      = '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',
            output_folder = config['OUTPUT_DIRECTORY'],
            save_to_disk  = False,
            mode          = config['STABLEDIFFUSION_MODE'],
            local_only    = False,
            torch_device  = config['TORCH_DEVICE'],
            sd_cache_dir  = config['STABLEDIFFUSION_CACHE_DIR'])

    def get_worker_method(self, worker:DeguDiffusionWorker):
        return worker.generate_image

    def get_config():
        return {
            'COMPACT_RESPONSES': False,
            'DEFAULT_IMAGES_PER_JOB': 8,
            'DEFAULT_PROMPT': 'Degu enjoys its morning coffee by {random_artists}, {random_tags}',
            'HUGGINGFACES_TOKEN': '',
            'IMAGES_HEIGHT': 512,
            'IMAGES_WIDTH': 512,
            'MAX_IMAGES_PER_JOB': 64,
            'MAX_IMAGES_BEFORE_THREAD': 2,
            'MAX_INFERENCES_PER_IMAGE': 120,
            'MAX_GUIDANCE_SCALE_PER_IMAGE': 30,
            'OUTPUT_DIRECTORY': 'images',
            'STABLEDIFFUSION_CACHE_DIR': 'stablediffusion_cache',
            'STABLEDIFFUSION_LOCAL_ONLY': False,
            'STABLEDIFFUSION_MODEL_NAME': 'CompVis/stable-diffusion-v1-4',
            'STABLEDIFFUSION_MODE': 'fp32',
            'SAVE_IMAGES_TO_DISK': True,
            'TORCH_DEVICE': 'cuda'
        }

    QUEUE = MyQueue(generate_worker, functools.partial(get_worker_method, get_config()))

    def image_response_for(image_data) -> str:
        image_data_base64 = base64.b64encode(image_data)
        image_data_header = "data:image/png;base64,".encode("ascii")
        image_data_complete = image_data_header + image_data_base64
        image_data_size = len(image_data_complete)
        image_data_digest = hashlib.sha256(image_data_complete).hexdigest()
        response = {
            "version": "0",
            "type": "@stablediffusion_txt2img_response",
            "size": image_data_size,
            "digest": image_data_digest,
            "data": image_data_complete.decode('ascii')
        }
        return json.dumps(response)

    async def echo(websocket):
        async for message in websocket:
            try:
                content = json.loads(message)
                print(content)
                
                QUEUE.add_job(Job(
                    external_reference=JobRequest(websocket, "5"),
                    iterations = 1,
                    kwargs = content
                ))

            except Exception as e:
                traceback.print_exception(e)
                await websocket.send(message)


    async def main():
        
        websocket_server = await serve(echo, "localhost", 8765)

        await asyncio.gather(websocket_server.serve_forever(), QUEUE.main_task())
        #async with serve(echo, "localhost", 8765):
              # run forever

    try:
        asyncio.run(main())
    except:
        QUEUE._bailing_out()
        exit(1)
    
