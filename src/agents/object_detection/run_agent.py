import argparse
import datetime
import os

from dotenv import load_dotenv, find_dotenv

from langchain_openai import AzureChatOpenAI

from agent import SystemAgent, SystemAgentState


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Object Detection agent. Traces are sent to Phoenix to localhost:6006."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="The temperature to use in the LLM. The higher the temperature, the more random the output. "
             "The default value is 0.001.",
        default=0.001
    )
    parser.add_argument(
        "--print-diagram",
        action="store_true",
        help="Print the diagram of the graph in mermaid format.",
        default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    temperature = args.temperature
    print_diagram = args.print_diagram

    # Build the run name, composed of the output directory and the current date
    run_name = f"Object Detection Agent-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    # Instrument with Phoenix --------------------------------------------------------
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register(
        project_name=run_name,
        endpoint="http://localhost:6006/v1/traces",
    )

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    # --------------------------------------------------------------------------------

    # Instantiate the LLM. Currently, use Azure OpenAI models
    # Read LLM configuration from environment variables
    _ = load_dotenv(find_dotenv())  # read local .env file
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    llm = AzureChatOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        azure_deployment=azure_openai_deployment,
        api_version=azure_openai_api_version,
        temperature=temperature
    )

    # Create the agent
    agent = SystemAgent(llm, "yolo11n.pt")

    if print_diagram:
        # Print the diagram of the graph
        mermaid_code = agent.compiled_graph.get_graph().draw_mermaid()
        ascii = agent.compiled_graph.get_graph().draw_ascii()
        print("\n\n\n------------------------------------------------------------------------------------------------")
        print("Mermaid code for the graph:")
        print("------------------------------------------------------------------------------------------------")
        print(mermaid_code)
        print("\n\n\n")
        print("------------------------------------------------------------------------------------------------")
        print("\n\n\n")
        print("ASCII code for the graph:")
        print("------------------------------------------------------------------------------------------------")
        print(ascii)
        print("\n\n\n")

    graph = agent.compiled_graph

    # Ask for the description of the diagram and read from the standard input
    image_path = input("Enter the image path: ")

    state = SystemAgentState({
        "image_path": image_path,
        "detected_objects": [],
        "history_messages": [],
        "human_feedback": None,
        "finished_detection": False
    })

    # Run the agent
    # Thread
    thread = {"configurable": {"thread_id": "1"}}
    while not state["finished_detection"]:
        for event in graph.stream(state, thread, stream_mode="values"):
            print(event)
            state = event
        # # Get user input
        # try:
        #     print("\n\n\n" + "-" * 80)
        #     user_input = input("¿Estás satisfecho con los objetos detectados?:")
        # except:
        #     user_input = "Error. Proceso finalizado."
        # graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

    print("\n\n\n" + "-" * 80)
    print("Finished graph")


if __name__ == "__main__":
    main()
