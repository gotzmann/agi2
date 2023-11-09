import torch

from team_code.generate import setup_model_and_tokenizer, generate_text, get_ppl

# --- MAIN ---

dialog = [
    [
        [
            { "type": "text", "content": "What is the largest ocean in the world?" }
        ]
    ],
    [
        [
            { "type": "image", "content": "00001.png" }, 
            { "type": "text", "content": "How many pears and apples are there in the picture?" }
        ]
    ],
    [
        [
            { "type": "text", "content": "What animal made that sound??" }, 
            { "type": "audio", "content": "00000.wav" }
        ]
    ],
    [
        [
            { "type": "text", "content": "How many continents does the Pacific Ocean border?" }
        ], 
        [
            { "type": "text", "content": "Is it the oldest of all existing oceans?" }
        ]
    ]
]

answers = [
    "The Pacific Ocean is the largest and deepest of the world ocean basins.",
    "There are four pears and one red apple on the picture.",
    "That is the dog barking.",
    "Pacific Ocean border four continents in total.",
    "Yes, it's the oldest ocean."
]

def main():

    print("\n=== START ===")

    models, tokenizer = setup_model_and_tokenizer()

    num = 0
    for batches in dialog:

        history = None
        response = None

        for query in batches:

            print("\n\n=====================================================================================")
            print("\n=== [ QUERY ] ===\n", query)

            if history != None:
                history = (history, response)

            response, history = generate_text(models, tokenizer, query, history)
            ppl = get_ppl(models, tokenizer, (query, answers[num]), (history, response))

            print("\n=== [ RESPONSE ] ===\n", response)
            print("\n=== [ PPL ] ===", ppl[0], "===")
            # print("\n === HISTORY ===\n", history)

            num = num + 1

    print("\n=== FINISH ===")

if __name__ == "__main__":
    main()

# --- END ---