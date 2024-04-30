import gradio as gr
from huggingface_hub import InferenceClient



if gr.NO_RELOAD:
    client = InferenceClient(token="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")

system_message = {
    "role": "system",
    "content": """
You are a helpful assistant.
You will be given a question and a set of answers along with a confidence score between 0 and 1 for each answer.
You job is to turn this information into a short, coherent response.
For example:
Question: "Who is being invoiced?", answer: {"answer": "John Doe", "confidence": 0.98}
You should respond with something like:
With a high degree of confidence, I can say John Doe is being invoiced.
Question: "What is the invoice total?", answer: [{"answer": "154.08", "confidence": 0.75}, {"answer": "155", "confidence": 0.25}
You should respond with something like:
I belive the invoice total is $154.08 thought it can also be $155.
"""}

def chat_fn(multimodal_message):
    question = multimodal_message["text"]
    image = multimodal_message["files"][0]

    answer = client.document_question_answering(
        image = image,
        question = question,
        model = "impira/layoutlm-document-qa"
    )

    answer = [{"answer": a.answer, "confidence": a.score} for a in answer]

    user_message = {"role": "user", "content": f"Question: {question}, answer: {answer}"}

    message = ""
    for token in client.chat_completion(messages = [system_message, user_message],
                                        max_tokens = 2000,
                                        stream = True,
                                        model = "HuggingFaceH4/zephyr-7b-beta"):
        if token.choices[0].finish_reason is not None:
            continue
        message += token.choices[0].delta.content
        yield message


with gr.Blocks() as demo:
    response = gr.Textbox(lines=5, label="Response")
    chat = gr.MultimodalTextbox(file_types=["image"], interactive=True,
                                show_label=False, placeholder="Upload an Image")
    chat.submit(chat_fn, chat, response)

if __name__ == "__main__":
    demo.launch()