import argparse
import json
from . import catalog

def main():
    parser=argparse.ArgumentParser(description='Inspect kernel assets; no HPU allocation or inference')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args();data=catalog()
    if args.json:
        print(json.dumps(data,indent=2));return
    print(f"Tensor runtime binding: {data['tensor_runtime_binding']}; no production backend is auto-selected.")
    for entry in data['families']:
        print(f"{entry['id']:<38} {entry['readiness']:<35} {entry['contract']}")

if __name__=='__main__':main()
