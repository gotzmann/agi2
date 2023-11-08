# from team_code.utils import get_query_from_input, get_text_emb
from team_code.generate import setup_model_and_tokenizer, generate_text

# --- MAIN ---

DIALOGUE_DICT = [
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
        {"type": "audio", "content": "path_to_the_audio"}
    ],
    [
        {"type": "image", "content": "path_to_the_image"}, 
        {"type": "text", "content": "What animal is on the picture?"}
    ]
]

def main():

    print("\n\n=== START ===")

    history_tensor = None
    models, tokenizer = setup_model_and_tokenizer()

    for cur_query_list in DIALOGUE_DICT:
        print("\n => cur_query_list = ", cur_query_list)
        generate_text(models, tokenizer, cur_query_list, history_tensor)

    print("\n\n=== FINISH ===")

if __name__ == "__main__":
    main()

# --- END ---