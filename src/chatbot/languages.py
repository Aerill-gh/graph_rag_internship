from typing import Literal


Language = Literal["en", "de", "lv", "es"]
DEFAULT_LANGUAGE: Language = "en"
LANGUAGE_NAMES: dict[Language, str] = {
    "en": "English",
    "de": "Deutsch",
    "lv": "Latviešu",
    "es": "Español",
}
LANGUAGE_INSTRUCTION_NAMES: dict[Language, str] = {
    "en": "English",
    "de": "German",
    "lv": "Latvian",
    "es": "Spanish",
}
NOT_FOUND_RESPONSES: dict[Language, str] = {
    "en": "I could not find enough information in the retrieved Zava documents to answer.",
    "de": "In den abgerufenen Zava-Dokumenten konnte ich nicht genügend Informationen finden, um die Frage zu beantworten.",
    "lv": "Atrastos Zava dokumentos neatradu pietiekami daudz informācijas, lai atbildētu uz jautājumu.",
    "es": "No encontré suficiente información en los documentos de Zava recuperados para responder.",
}