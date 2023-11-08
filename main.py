import torch

from team_code.generate import setup_model_and_tokenizer, generate_text

# --- MAIN ---

dialog = [
    [
        {"type": "image", "content": "00001.png"}, 
        {"type": "text", "content": "What animal is on the picture?"}
    ],
    [
        {"type": "text", "content": "What is the largest ocean in the world?"}
    ], 
    [
        {"type": "text", "content": "How many continents does the Pacific Ocean border?"}
    ], 
    [
        {"type": "text", "content": "Is it the oldest of all existing oceans?"}
    ],
    [
        {"type": "text", "content": "What animal made that sound??"}, 
        {"type": "audio", "content": "0000.wav"}
    ]
]

def main():

    print("\n=== START ===")

    models, tokenizer = setup_model_and_tokenizer()

    history = None
    response = None

    for query in dialog:
        print("\n => cur_query_list = ", query)
        tmp = (history, response)
        response, history = generate_text(models, tokenizer, query, torch.tensor(tmp))
        print("\n === RESPONSE ===\n\n", response)
        print("\n === HISTORY SIZE ===\n\n", history.Size())
        print("\n === HISTORY ===\n\n", history)

    print("\n=== FINISH ===")

if __name__ == "__main__":
    main()

# --- END ---