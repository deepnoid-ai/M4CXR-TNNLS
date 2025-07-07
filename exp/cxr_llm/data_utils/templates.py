# system messages
SYSTEM_BASE = "The following is a conversation between a curious human and an AI medical assistant."
SYSTEM_DETAIL = (
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SYSTEM_MESSAGE = SYSTEM_BASE + " " + SYSTEM_DETAIL

# special media tokens
IMAGE = "<image>"

MEDIA_TOKENS = {
    "image": [IMAGE],
}

# constants
IGNORE_INDEX = -100  # default ignore index of CrossEntropyLoss

# mistral chat template
MISTRAL_USER = "[INST]"
MISTRAL_ASSISTANT = "[/INST]"
MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

# gemma chat template
GEMMA_USER = "<start_of_turn>user\n"
GEMMA_ASSISTANT = "<start_of_turn>model\n"
GEMMA_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

# vicuna chat template
VICUNA_USER = "\nUSER: "
VICUNA_ASSISTANT = "\nASSISTANT: "
VICUNA_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '\nUSER: ' + message['content'] | trim}}{% elif message['role'] == 'assistant' %}{{ '\nASSISTANT: ' + message['content'] | trim + eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\nASSISTANT: ' }}{% endif %}"

###############################################################################
# default patterns
###############################################################################

# pattern name
# list => random selection
# (system prompt, images, question, answer, ...)

default_template = {
    # report only
    "report_only": [
        ("", "<image>", "", "{report}"),
    ],
    # medical report generation
    "mrg": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide a description of the findings in the radiology image.",
            "{report}",
        )
    ],
    # medical report generation with indication
    "mrg_w_i": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide a description of the findings in the radiology image given the following indication: {indication}",
            "{report}",
        )
    ],
    # single disease
    "single_disease": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Is {finding} present in the radiology image?",
            "Yes.",
            "No.",
            "Uncertain.",
        )
    ],
    "single_disease_no_finding": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Are there any findings in the radiology image?",  # question conversely
            "Yes.",
            "No.",
            "Uncertain.",
        )
    ],
    # multi disease
    "multi_disease": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Which of the following findings are present in the radiology image? Findings: {findings}",
            "{positive_findings}.",  # positive
            "Not found.",  # negative
        )
    ],
    # chain-of-thought (multi disease & report generation)
    "cot": [
        (
            "",
            "",
            "Based on the previous conversation, provide a description of the findings in the radiology image.",
            "{report}",
        )
    ],
    # chain-of-thought (multi disease & report generation)
    "frontal_lateral_cot": [
        (
            SYSTEM_MESSAGE,
            "radiology images: {images}",
            "Which of the following findings are present in the radiology images? Findings: {findings}",
            "{positive_findings}.",  # positive
            "Not found.",  # negative
            # 2nd QA
            "Based on the previous conversation, provide a description of the findings in the radiology images.",
            "{report}",
        )
    ],
    # chain-of-thought (multi disease & report generation)
    "history_cot": [
        (
            SYSTEM_MESSAGE,
            "prior radiology images: {prior_images}, prior radiology report: {prior_findings}",
            "follow-up images: {images}, The radiology studies are given in chronological order. Which of the following findings are present in the current follow-up radiology images? Findings: {findings}",
            "{positive_findings}.",  # positive
            "Not found.",  # negative
            "Based on the previous conversation, provide a description of the findings in the current follow-up radiology images.",
            "{report}",
        ),
    ],
    # VQA
    "vqa": [
        (
            SYSTEM_MESSAGE,
            "radiology image: {image}",
            "Answer the question. {question}",
            "{answer}",
        ),
    ],
    # Radialog VQA
    "radialog_vqa": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "{question}",
            "{answer}",
        ),
    ],
    # Difference VQA
    "diff": [
        (
            SYSTEM_MESSAGE,
            "reference: <image>, main: <image>",
            "Using the provided reference and main radiology images, answer the following question. {question}",
            "{answer}",
        ),
    ],
    # finding grounding
    "f_ground": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Is {finding} present in the radiology image? If so, provide the bounding box coordinates of the region.",
            "Yes. {bbox}.",  # positive
            "Not present.",  # negative
        ),
    ],
    # grounded finding
    "grounded_f": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide a finding name for this region. {bbox}",
            "{finding}",
        ),
    ],
    # multi finding grounding
    "mf_ground": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Which of the following findings are present in the radiology image? Provide the bounding box coordinates if present. Findings: {findings}",
            "{findings_bboxes}.",  # positive (consolidataion [0,1,0,3], atelectasis [2,3,2,5], [3,6,7,8], ...)
            "Not found.",  # negative
        ),
    ],
    # abnormality detection
    "abn_det": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide the bounding box coordinates of abnormal regions in the radiology image.",
            "{bbox}.",
        ),
    ],
    # organ grounding
    "o_ground": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide the bounding box coordinates of {organ} in the radiology image.",
            "{bbox}",
        ),
    ],
    # grounded organ
    "grounded_o": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide an organ name for this region. {bbox}",
            "{organ}",
        ),
    ],
    # phrase grounding
    "p_ground": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide the bounding box coordinate of the region this phrase describes: {phrase}",
            "{bbox}.",
        ),
    ],
    # grounded phrase
    "grounded_p": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide a radiology report phrase for the region. {bbox}.",
            "{phrase}.",
        ),
    ],
    # anatomical grounding
    "a_ground": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide the bounding box coordinate of the anatomical region. {name}",
            "{bbox}.",
        ),
    ],
    # grounded anatomical
    "grounded_a": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide an anatomical region name for this region. {bbox}",
            "{name}.",
        ),
    ],
    # General medical QA
    "qa": [
        (
            SYSTEM_MESSAGE,
            "",
            "instruction : {instruction}, question : {question}",
            "{answer}",
        ),
    ],
    "pubmed_closed-qa": [
        (
            SYSTEM_MESSAGE,
            "",
            "instruction : Answer the following question based on the given sentence. {question}, sentence : {sentence}",
            "{answer}",
        ),
    ],
    "closed-qa": [
        (
            SYSTEM_MESSAGE,
            "",
            "instruction : {instruction}, {question}",
            "{answer}",
        ),
    ],
    "cord19_qa_summarization": [
        (
            SYSTEM_MESSAGE,
            "",
            "instruction : {instruction}: {question}",
            "{answer}",
        ),
    ],
    "usmle_self_assessment_closed-qa": [
        (
            SYSTEM_MESSAGE,
            "",
            "instruction : Answer with the option's letter from the given choices directly. {question} There are several options : {option}",
            "{answer}",
        ),
    ],
    "qa-cot": [
        (
            "",
            "",
            "Based on the previous Q&A conversation, summarize the provided answer.",
            "{answer}",
        ),
    ],
    "frontal": [
        (
            SYSTEM_MESSAGE,
            "radiology image: <image>",
            "Provide a description of the findings in the radiology image.",
            "{report}",
        ),
    ],
    "frontal_lateral": [
        (
            SYSTEM_MESSAGE,
            "radiology images: {image}",
            "Provide a description of the findings in the radiology images.",
            "{report}",
        ),
    ],
    "finding_to_impression": [
        (
            SYSTEM_MESSAGE,
            "",
            "The 'Findings' section of a chest x-ray radiology report is given: {findings} Please write the 'Impression' section based on the 'Findings' section.",
            "{impression}",
        ),
    ],
    "history": [
        (
            SYSTEM_MESSAGE,
            "prior radiology images: {prior_images}, prior radiology report: {prior_findings}",
            "follow-up images: {images}, The radiology studies are given in chronological order. Provide a description of the findings in the current follow-up radiology images.",
            "{report}",
        ),
    ],
    "history_all": [
        (
            SYSTEM_MESSAGE,
            "prior radiology images: {prior_images}, current radiology images: {current_images}",
            "The radiology images are given in chronological order. Analyze the progression and provide a description of the findings in the current radiology images.",
            "{report}",
        ),
    ],
}
