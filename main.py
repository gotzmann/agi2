import torch

from team_code.generate import setup_model_and_tokenizer, generate_text

# --- MAIN ---

dialog = [
    [
        { "type": "image", "content": "00001.png" }, 
        { "type": "text", "content": "What animal is on the picture?" }
    ],
    [
        { "type": "text", "content": "What animal made that sound??" }, 
        { "type": "audio", "content": "00000.wav" }
    ],
    [
        { "type": "text", "content": "What is the largest ocean in the world?" }
    ], 
    [
        { "type": "text", "content": "How many continents does the Pacific Ocean border?" }
    ], 
    [
        { "type": "text", "content": "Is it the oldest of all existing oceans?" }
    ]
]

def main():

    print("\n=== START ===")

    models, tokenizer = setup_model_and_tokenizer()

    history = None
    response = None

    for query in dialog:

        print("\n=> query = ", query)

        #tupl = None
        if history != None and response != None:
            history = (history, response)
            # tmp = torch.tensor(tmp)

        response, history = generate_text(models, tokenizer, query, history)

        print("\n === RESPONSE ===\n", response)
        print("\n === HISTORY SIZE ===\n", len(history))
        print("\n === HISTORY ===\n", history)

    print("\n=== FINISH ===")

if __name__ == "__main__":
    main()

# --- END ---