openai_api_key = "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"
import time
import numpy
import io
import soundfile as sf
import sounddevice as sd
from fuzzywuzzy import fuzz
from openai import OpenAI


# whisper(user audio) -> query
# llama(query) -> completion
# loop until k-token convergence
# gpt(completion) -> response
# tts(response) -> response audio


client = OpenAI(api_key=openai_api_key)

query_completions = []


def find_converged_completion(query):
    # Given an incomplete query, complete it. When a prior completion matches the last k tokens of the user's query, return it.
    for old_query, completion in query_completions[::-1]:
        print('Check:')
        print(query)
        print('')
        print('Completion:')
        print(completion)
        print('')
        print('-' * 20)
        print('')
        # print('CHECK', query, '|||||||', completion)
        if check_match(completion, query):
            return " ".join([old_query, completion])

    completion = llm(query)
    if completion.startswith(query):
        completion = completion.replace(query, "")

    query_completions.append((query, completion))
    return None


def check_match(completion, query, k=3, fuzz_threshold=75):
    # Given a completion and a query, check if the last k tokens of the query match the completion. If so, return True.
    query_toks = query.split(" ")[-k:]
    completion_toks = completion.split(" ")[:k]

    query_str = " ".join(query_toks)
    completion_str = " ".join(completion_toks)

    if fuzz.ratio(query_str, completion_str) > fuzz_threshold:
        return True
    else:
        return False


def llm(query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Please complete the user's text. Do not repeat the user's text. Do not add or say anything else. Simply respond by 'autocompleting' their text. If they've completed a sentence, continue with the next sentence.",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=20,
    )

    print('COMPLETION: ', response.choices[0].message.content)

    return response.choices[0].message.content


def gpt(completion):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": completion},
        ],
    )

    print('INTERRUPTION: ', response.choices[0].message.content)

    return response.choices[0].message.content


def tts(response):
    spoken_response = client.audio.speech.create(
        model="tts-1-hd", voice="fable", response_format="opus", input=response
    )

    buffer = io.BytesIO()
    for chunk in spoken_response.iter_bytes(chunk_size=4096):
        buffer.write(chunk)
    buffer.seek(0)

    with sf.SoundFile(buffer, "r") as sound_file:
        data = sound_file.read(dtype="int16")
        sd.play(data, sound_file.samplerate)
        sd.wait()


def thoughtocomplete(query):
    # print('THOUGHTOCOMPLETE', query)
    completion = find_converged_completion(query)
    if completion:
        response = gpt(completion)
        # print(response)
        tts(response)
        return True


if __name__ == "__main__":
    query = "What is the following song? Big wheels keep on turning. Carry me home to see my kin. Singing songs about the south land."
    query_toks = query.split(" ")

    for i in range(1, len(query_toks)):
        curr_query = " ".join(query_toks[:i])
        print(curr_query)
        done = thoughtocomplete(curr_query)
        if done:
            break
        # time.sleep(1)

    # tts("Hello, my name is Chat G P T. I am a helpful assistant.")