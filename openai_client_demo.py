from openai import OpenAI

c = OpenAI(base_url="https://k050506koch-gpt3-dev-api.hf.space/v1")

print(c.models.list())


prompt="He is a doctor. His main goal is"
a1 = c.completions.create(model="GPT3-dev-350m-2805", prompt=prompt, max_tokens=64)
print("\n\n", a1)
print("\n\n", prompt + a1.choices[0].text)