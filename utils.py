import openai
from time import sleep
import logging

def generate(self, query_prompt):
    complete = False
    ntries = 0
    while not complete:
        try:
            raw_responses = openai.Completion.create(
                model=self.engine,
                prompt=query_prompt,
                temperature=self.temp,
                max_tokens=self.max_tokens,
                stop=self.stop,
                n=self.n,
                # logprobs=5
            )
            complete = True
        except:
            sleep(30)
            logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
            logging.info("OK continue")
            ntries += 1
    if self.n == 1:
        responses = [raw_responses["choices"][0]["text"].strip()]
    else:
        responses = [choice["text"].strip() for choice in raw_responses["choices"]]
    return responses