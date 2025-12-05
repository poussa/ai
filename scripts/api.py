#!/usr/bin/env python3

#
# Examples:
#   python api.py --port 8080 --path /v1 --api oa
#   python api.py --port 8080 --path /v1 --api hf --no-stream
#

import requests
from requests.exceptions import HTTPError
import argparse
import json
from urllib.parse import urlunsplit
from huggingface_hub import InferenceClient
from openai import OpenAI

#
# Huggingface API
#
def hf_chat_api(url, args):

    client = InferenceClient(
        base_url=url,
    )

    output = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "AI assistant."},
            {"role": "user", "content": "Whats is the capital of Finland?"},
        ],
        stream=args.stream,
        max_tokens=1024,
    )
    if args.stream:
        for chunk in output:
            print(chunk.choices[0].delta.content)
    else:
        print(output.choices[0].message.content)
    return

#
# OpeaAI API
#
def oa_chat_api(url, args):

    client = OpenAI(
        base_url=url,
        api_key="-"
    )
    for i in range(args.count):
        output = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "AI assistant."},
                {"role": "user", "content": args.prompt},
            ],
            stream=args.stream,
            temperature=args.temperature
        )

        if args.stream:
            for message in output:
                print(message.choices[0].delta.content, end='')
        else:
            print(output.choices[0].message.content)

#
# main
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='api', description='API calls to various endpoints')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--scheme', type=str, default="http")
    parser.add_argument('--host', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--api', type=str, default="oa", choices=["hf", "oa"])
    parser.add_argument('--prompt', type=str, default="Tell me a joke!")
    parser.add_argument('--stream', type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--headers', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--json', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--verify', type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--count', type=int, default=1)
    args = parser.parse_args()

    url = urlunsplit((args.scheme, f"{args.host}:{args.port}", args.path, "", ""))
    print(url)
    
    match args.api:
        case "hf":
            hf_chat_api(url, args)
        case "oa":
            oa_chat_api(url, args)
        case _:
            parser.print_help()
