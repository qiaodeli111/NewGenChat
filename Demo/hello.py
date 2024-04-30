import gradio as gr

def greet(name, lastname, intensity):
    return "Hello, " + name +  "!" * int(intensity)

# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "text", "slider"],
#     outputs=["text"],
# )


demo = gr.Interface(lambda name: name, gr.MultimodalTextbox(), "text")

demo.launch()
