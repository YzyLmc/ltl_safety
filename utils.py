import openai
from time import sleep
import logging

class GPT4:
    def __init__(self, engine="gpt-4", temp=0, max_tokens=128, n=1, stop=['\n']):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    def generate(self, query_prompt):
        complete = False
        ntries = 0
        while not complete:
            try:
                raw_responses = openai.ChatCompletion.create(
                    model=self.engine,
                    messages=prompt2msg(query_prompt),
                    temperature=self.temp,
                    n=self.n,
                    stop=self.stop,
                    max_tokens=self.max_tokens,
                )
                complete = True
            except:
                sleep(30)
                logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...")
                # logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
                logging.info("OK continue")
                ntries += 1
        if self.n == 1:
            responses = [raw_responses["choices"][0]["message"]["content"].strip()]
        else:
            responses = [choice["message"]["content"].strip() for choice in raw_responses["choices"]]
        return responses
