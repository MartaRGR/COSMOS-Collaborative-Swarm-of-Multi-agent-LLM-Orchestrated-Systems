import json
import pandas as pd
from torch.backends.cudnn import deterministic


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return json.load(f)


def extract_final_answer_letter(text):
    if not text:
        return None

    import re
    # Buscar patrón <<< FINAL ANSWER: X >>>
    match = re.search(r'<<< FINAL ANSWER:\s*([A-E])\s*>>>', str(text))
    if match:
        return match.group(1)

    # Si no encuentra el patrón, buscar solo letras A-E al final
    match = re.search(r'\b([A-E])\b(?=\s*$)', str(text))
    if match:
        return match.group(1)

    return None


def correct_answer(root_folder, dataset_answer, question_id):
    if root_folder == 'logic_math_responses':
        folder_path = 'logic_math_dataset'
    else:
        folder_path = ''
    response_file = read_json_file(f'{folder_path}/{dataset_answer}.json')
    for item in response_file:
        if item['id'] == question_id:
            return item['label']
    return None


def process_dataset(dataset_answers, dataset_responses, deterministic, root_folder='logic_math_responses'):
    json_file = f'{root_folder}/{dataset_responses}{"_deterministic" if deterministic else ""}_overall_state.json'
    data = read_json_file(json_file)

    # Procesar cada elemento en la lista
    records = []
    for item in data:
        try:
            # Extraer información básica
            task_id = item.get('task_id')
            user_task = item.get('user_task', '')

            # Extraer la pregunta del user_task
            question = None
            if 'Question:' in user_task:
                question_part = user_task.split('Question:')[1]
                if 'Options:' in question_part:
                    question = question_part.split('Options:')[0].strip()
                else:
                    question = question_part.strip()
            elif user_task:
                # Si no hay "Question:", usar el user_task completo
                question = user_task.strip()

            # Obtener resultados de agregación
            answers = item.get('answer', [])
            agg_results = {}

            for answer in answers:
                method = answer.get('aggregation_method')
                if method in ['llm_synthesis', 'majority_vote', 'weighted_average']:
                    result = answer.get('result', {})
                    if isinstance(result, dict):
                        response = result.get('response', {})
                        if isinstance(response, dict):
                            final_answer = response.get('final_answer')
                            if final_answer is None:
                                # Buscar alternativas
                                final_answer = response.get('final_aggregated_count')
                            agg_results[method] = final_answer
                        else:
                            agg_results[method] = response

            # Procesar cada crew
            crews_plan = item.get('crews_plan', [])

            for crew_info in crews_plan:
                crew_name = crew_info.get('name')

                # Buscar en task_plan
                task_plan = crew_info.get('task_plan', {})
                tasks = task_plan.get('tasks', [])

                for task in tasks:
                    subtasks = task.get('subtasks', {})
                    for subtask_key, subtask_list in subtasks.items():
                        for subtask in subtask_list:
                            agent = subtask.get('agent', {})

                            # Extraer información del agente
                            model = agent.get('model')
                            hyperparameters = agent.get('hyperparameters', {})
                            crew_response = agent.get('result')

                            # Extraer solo la letra de FINAL ANSWER si existe
                            final_answer_letter = extract_final_answer_letter(crew_response)

                            # Crear registro
                            record = {
                                'id': task_id,
                                'question': question,
                                'crew': crew_name,
                                'modelo': model,
                                'hyperparameters': json.dumps(hyperparameters) if hyperparameters else None,
                                # 'crew_response': crew_response,
                                'crew_response': final_answer_letter,
                                'llm_synthesis': agg_results.get('llm_synthesis'),
                                'weighted_average': agg_results.get('weighted_average'),
                                'majority_vote': agg_results.get('majority_vote'),
                                'correct_answer': correct_answer(root_folder, dataset_answers, task_id)
                            }

                            records.append(record)

        except Exception as e:
            print(f"Error procesando elemento: {e}")
            continue

    return records

deterministic = False # If fixed crews are used
for dataset in [{'questions': 'mmlu-math_v2', 'responses': 'mmlu-math_v2_deterministic'}]: #  ['aqua-rat', 'gsm8k', 'logiqa', 'mmlu', 'svamp']
    records = process_dataset(dataset["questions"], dataset["responses"], deterministic)

    # Crear DataFrame
    df = pd.DataFrame.from_records(records)

    # Eliminar duplicados si los hay (misma pregunta y crew)
    df = df.drop_duplicates(subset=['id', 'crew'], keep='first')

    print(f"Se procesaron {len(df)} registros")
    print("\nPrimeras 5 filas:")
    print(df.head())

    # Opcional: guardar en CSV
    if deterministic:
        dataset += '_deterministic'
    df.to_csv(f'{dataset["responses"]}_responses.csv', index=False)