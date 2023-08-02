import os
import time
import openai
import json
import datetime
import numpy as np

class LLMPrompter():
    def __init__(self, gpt_version, api_key) -> None:
        self.gpt_version = gpt_version
        if api_key is None:
            raise ValueError("OpenAI API key is not provided.")
        else:
            openai.api_key = api_key

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str) -> str:
        while True:
            try:
                if 'gpt-4' in self.gpt_version:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[
                                {"role": "system", "content": prompt['system']},
                                {"role": "user", "content": prompt['user']},
                            ],
                        **sampling_params
                        )
                    # print("response: ", response)
                else:
                    response = openai.Completion.create(
                        model=self.gpt_version,
                        prompt=prompt,
                        **sampling_params
                    )
            except Exception as e:
                print("Request failed, sleep 2 secs and try again...", e)
                time.sleep(2)
                continue
            break

        if save:
            key = self.make_key()
            output = {}
            os.system('mkdir -p {}'.format(save_dir))
            if os.path.exists(os.path.join(save_dir, 'response.json')):
                with open(os.path.join(save_dir, 'response.json'), 'r') as f:
                    prev_response = json.load(f)
                    output = prev_response

            with open(os.path.join(save_dir, 'response.json'), 'w') as f:
                if 'gpt-4' in self.gpt_version:
                    output[key] = {
                                'prompt': prompt,
                                'sampling_params': sampling_params,
                                'response': response['choices'][0]['message']["content"].strip()
                            }
                else:
                    output[key] = {
                                'prompt': prompt,
                                'sampling_params': sampling_params,
                                'response': response['choices'][0]['text'].strip(),
                                'logprob': np.mean(response['choices'][0]['logprobs']['token_logprobs'])
                            }
                json.dump(output, f, indent=4)
            
        if 'gpt-4' in self.gpt_version:
            return response['choices'][0]['message']["content"].strip(), None
        else:
            return response['choices'][0]['text'].strip(), np.mean(response['choices'][0]['logprobs']['token_logprobs'])

    def make_key(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")