from langchain.agents import initialize_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.llms.baseten import LLM
from langchain.agents import AgentType
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union
import openpyxl, tqdm, time, requests
from typing import Optional, List, Dict, Mapping, Any
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from langchain_community.llms import OpenAI
from langchain_community.llms import ChatGLM
import os, json, torch, openai
import fastchat.model

os.environ["OPENAI_API_KEY"] = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type="openai"
openai.base_url="https://us.claudeshop.top/v1"
os.environ["OPENAI_BASE_URL"]="https://us.claudeshop.top/v1"


class ChatGPT():

    def __init__(self):
        self.client = openai.OpenAI()
       # print(os.getenv("OPENAI_API_KEY"))

    def __call__(self, prompt, history=[], temperature=0.8) -> str:
        mes = [{"role": "system", "content": "You are a helpful assistant."}]
        for h in history:
            mes.append({"role": "user", "content": h[0]})
            mes.append({"role": "assistant", "content": h[1]})
        mes.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=mes,
            temperature=temperature
        )
        response = completion.choices[0].message.content

        return response


class Claude(LLM):

    url:str = 'http://ecs.sv.us.alles-apin.openxlab.org.cn/v1/claude/v1/text/chat'

    @property
    def _llm_type(self) -> str:
        return "Claude"

    def _construct_query(self, prompt: str) -> Dict:

        query = {
            "prompt": prompt,
            "model": "claude-1",
            "max_tokens": 500
        }
        return query

    @classmethod
    def _post(cls, url: str, query: Dict) -> Any:

        _headers = {'Content_Type': 'application/json',
                    'alles-apin-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjozOSwidXNlcm5hbWUiOiJ0ZW5neWFuIiwiYXBwbHlfYXQiOjE2OTI5NTA0NzM3OTcsImV4cCI6MTY5NjA2MDg2OTM4NH0.mJwsjncd23JSTlFdFU16gcFzsrTiqMRG63nsb3Xz7jA'
                    }
        with requests.session() as sess:
            resp = sess.post(url,
                             data=json.dumps(query),
                             headers=_headers
                             # timeout=60
                             )
        return resp

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """_call
        """
        # construct query
        query = self._construct_query(prompt=prompt)

        # post
        while True:
            resp = self._post(url=self.url, query=query)

            if resp.status_code == 200:
                resp_json = resp.json()
                if resp_json["msg"] == 'ok':
                    print(resp_json)
                    predictions = resp_json["data"]["completion"]
                    return predictions
            else:
                return "Request model!"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {
            "url": self.url
        }
        return _param_dict


class ChatGLM2_6B():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
        self.model = self.model.eval()

    def __call__(self, prompt, history=[]) -> str:
        response, history = self.model.chat(self.tokenizer, prompt, history=[])

        return response



class InternLM_20B():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-20b", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-20b",
                                                          # device_map= 'balanced', load_in_8bit = False, max_memory = max_memory_mapping,
                                                          trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def __call__(self, prompt, history=[]) -> str:
        response, history = self.model.chat(self.tokenizer, prompt, temperature=0.8, history=history)

        return response


class InternLM_7B():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b", trust_remote_code=True).cuda()
        self.model = self.model.eval()

    def __call__(self, prompt, history=[]) -> str:
        response, history = self.model.chat(self.tokenizer, prompt, temperature=0.8, history=history)

        return response


class Baichuan2_13B():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", use_fast=False,
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", device_map="auto",
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat")

    def __call__(self, prompt, history=[]) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)

        return response


class Baichuan2_7B():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False,
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", device_map="auto",
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")

    def __call__(self, prompt, history=[]) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)

        return response


class Vicuna_7B():

    def __init__(self):

        self.model, self.tokenizer = fastchat.model.load_model(
            "lmsys/vicuna-7b-v1.5",
            device="cuda",
            num_gpus=1,
            max_gpu_memory="40G",
            dtype="auto",
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )

    def __call__(self, input_prompt, history=[]) -> str:

        conv = fastchat.model.get_conversation_template("vicuna-7b-v1.5")
        conv.append_message(conv.roles[0], input_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.8,
            max_new_tokens=256,
        )
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = self.tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
                output = output.strip()

        return output


class Vicuna_13B():

    def __init__(self):

        self.model, self.tokenizer = fastchat.model.load_model(
            "lmsys/vicuna-13b-v1.5",
            device="cuda",
            num_gpus=4,
            max_gpu_memory="40.0GB",
            dtype="auto",
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )

    def __call__(self, input_prompt, history=[]) -> str:

        conv = fastchat.model.get_conversation_template("vicuna-13b-v1.5")
        conv.append_message(conv.roles[0], input_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.8,
            max_new_tokens=256,
        )
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = self.tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
                output = output.strip()

        return output


class Vicuna_33B():

    def __init__(self):

        self.model, self.tokenizer = fastchat.model.load_model(
            "lmsys/vicuna-33b-v1.3",
            device="cuda",
            num_gpus=4,
            max_gpu_memory="40GB",
            dtype="auto",
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )

    def __call__(self, input_prompt, history=[]) -> str:

        conv = fastchat.model.get_conversation_template("vicuna-33b-v1.3")
        conv.append_message(conv.roles[0], input_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.8,
            max_new_tokens=256,
        )
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = self.tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
                output = output.strip()

        return output


class Qwen_7B():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto",
                                                          trust_remote_code=True).bfloat16().eval()

    def __call__(self, prompt, history=[]) -> str:
        response, history = self.model.chat(self.tokenizer, prompt, history=history)

        return response


class Qwen_14B():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-Chat", device_map="auto",
                                                          trust_remote_code=True).bfloat16().eval()

    def __call__(self, prompt, history=[]) -> str:
        response, history = self.model.chat(self.tokenizer, prompt, temperature=0.8, history=[])

        return response
