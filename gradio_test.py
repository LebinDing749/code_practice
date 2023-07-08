import gradio as gr


def greet(name):
    return "hello" + name + "!"


def greet2(name1, name2):
    return "hello" + name1 + ", " + name2 + "!"


demo = gr.Interface(fn=greet,
                    inputs='text',
                    outputs='text')

demo2 = gr.Interface(fn=greet2,
                     inputs=[gr.components.Textbox(label="name1"), gr.components.Textbox(label="name2")],
                     outputs='text')

combined_demo = gr.TabbedInterface([demo, demo2])
combined_demo.launch()