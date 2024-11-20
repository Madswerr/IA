import pandas as pd

# Listas de preguntas y respuestas (agregar más preguntas según sea necesario)
conceptos_basicos_ai = [
    ("¿Qué es un asistente virtual?", "Un asistente virtual es un software diseñado para ayudar a los usuarios a realizar tareas a través de comandos de voz o texto."),
    ("¿Cómo puedo usar un asistente virtual?", "Puedes interactuar con un asistente virtual utilizando comandos de voz o texto para solicitar información, programar recordatorios y realizar otras tareas."),
    ("¿Qué es un algoritmo?", "Un algoritmo es un conjunto de instrucciones o reglas definidas para resolver un problema o realizar una tarea específica."),
    ("¿Qué es la automatización?", "La automatización es el uso de tecnología para realizar tareas con mínima intervención humana."),
    ("¿Qué es la programación?", "La programación es el proceso de escribir instrucciones en un lenguaje de computadora para que realice tareas específicas."),
    ("¿Qué son los datos?", "Los datos son hechos y estadísticas recopilados para referencia o análisis."),
    ("¿Qué es una base de datos?", "Una base de datos es una colección organizada de datos que permite el almacenamiento, la gestión y la recuperación eficiente de información."),
    ("¿Qué es la ciberseguridad?", "La ciberseguridad se refiere a las prácticas y tecnologías diseñadas para proteger sistemas, redes y datos de ataques cibernéticos."),
    ("¿Qué es la computación en la nube?", "La computación en la nube es el uso de servidores remotos en Internet para almacenar, gestionar y procesar datos en lugar de hacerlo en un servidor local."),
    ("¿Qué son las redes sociales?", "Las redes sociales son plataformas en línea que permiten a los usuarios crear contenido, compartir información y conectarse con otros."),
    ("¿Qué es el Internet de las cosas (IoT)?", "El Internet de las cosas es una red de dispositivos físicos que están conectados a Internet y pueden intercambiar datos entre sí."),
    ("¿Qué es un virus informático?", "Un virus informático es un tipo de software malicioso que se replica a sí mismo y puede dañar archivos y sistemas en computadoras."),
    ("¿Qué es la realidad aumentada?", "La realidad aumentada es una tecnología que superpone información digital, como imágenes o datos, en el mundo real a través de dispositivos como teléfonos inteligentes o gafas especiales."),
    ("¿Qué es un sistema operativo?", "Un sistema operativo es el software que gestiona el hardware y software de una computadora, permitiendo que otras aplicaciones se ejecuten."),
    ("¿Qué es la programación orientada a objetos?", "La programación orientada a objetos es un paradigma de programación que utiliza 'objetos' que contienen datos y funciones para diseñar aplicaciones."),
    ("¿Qué es un motor de búsqueda?", "Un motor de búsqueda es una herramienta en línea que permite a los usuarios buscar información en Internet a través de palabras clave."),
    ("¿Qué son los dispositivos móviles?", "Los dispositivos móviles son dispositivos portátiles que pueden conectarse a Internet, como teléfonos inteligentes y tabletas."),
]

aprendizaje_automatico = [
    ("¿Qué es el aprendizaje automático?", "El aprendizaje automático es una subcategoría de la inteligencia artificial que utiliza algoritmos para permitir a las computadoras aprender de los datos."),
    ("¿Qué son los datos de entrenamiento en el aprendizaje automático?", "Los datos de entrenamiento son conjuntos de datos utilizados para entrenar modelos de aprendizaje automático."),
]

redes_neuronales = [
    ("¿Qué es una red neuronal?", "Una red neuronal es un modelo computacional inspirado en el funcionamiento del cerebro humano, utilizado para reconocer patrones y resolver problemas complejos."),
    ("¿Cuál es la función de activación en una red neuronal?", "La función de activación determina si una neurona se activará o no, transformando la suma ponderada de las entradas en una salida que puede ser utilizada en la siguiente capa de la red."),
]

saludos_y_conversaciones = [
    ("¿Cómo te llamas?", "Soy un asistente virtual diseñado para ayudarte con tus preguntas."),
    ("¿Cómo estás?", "Estoy aquí para ayudarte. ¿En qué puedo asistirte hoy?"),
    ("¿Qué puedes hacer?", "Puedo responder a tus preguntas, ayudarte con información y realizar tareas básicas."),
    ("¿Cuál es tu función?", "Mi función es proporcionar información y apoyo en diversas áreas."),
    ("¿Puedes hablar de temas específicos?", "Sí, puedo conversar sobre una variedad de temas, como tecnología, ciencia y más."),
    ("¿Cómo puedo mejorar mis habilidades de conversación?", "Practicar la conversación con otros y leer más sobre los temas que te interesan puede ayudar."),
    ("¿Qué es un chatbot?", "Un chatbot es un programa de computadora diseñado para simular una conversación humana."),
    ("¿Cuáles son tus limitaciones?", "No tengo emociones y mi conocimiento está basado en la información con la que fui entrenado."),
    ("¿Qué es la inteligencia artificial?", "La inteligencia artificial es el campo que se ocupa de crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana."),
    ("¿Puedes darme un consejo?", "Claro, lo mejor es siempre escuchar atentamente y mostrar interés genuino en lo que dice la otra persona."),
    ("¿Por qué es importante la comunicación?", "La comunicación es clave para entendernos y construir relaciones efectivas."),
    ("¿Cuál es la mejor manera de iniciar una conversación?", "Un buen inicio puede ser un saludo y una pregunta abierta sobre el bienestar de la otra persona."),
    ("¿Cómo puedo responder a una queja?", "Es importante escuchar con atención y mostrar empatía hacia la persona que se queja."),
    ("¿Qué temas son buenos para conversaciones casuales?", "Temas como el clima, películas, música o deportes suelen ser buenos para comenzar una conversación."),
    ("¿Cómo puedo despedirme educadamente?", "Puedes decir 'Fue un placer hablar contigo' o 'Hasta luego, que tengas un buen día'."),
    ("¿Qué son las habilidades sociales?", "Las habilidades sociales son las capacidades que nos permiten interactuar y comunicarnos efectivamente con los demás."),
    ("¿Cómo puedo mejorar mis habilidades sociales?", "Puedes practicar interactuando con diferentes personas y observando cómo se comunican."),
]

base_conocimiento_ai = [
    ("¿Qué es la inteligencia artificial?", "La inteligencia artificial es el campo de estudio que se ocupa de la creación de sistemas capaces de realizar tareas que normalmente requieren inteligencia humana."),
    ("¿Qué es el aprendizaje automático?", "El aprendizaje automático es una subcategoría de la inteligencia artificial que permite a las máquinas aprender de los datos y mejorar su rendimiento con el tiempo."),
    ("¿Cuáles son los tipos de aprendizaje automático?", "Los tipos principales son el aprendizaje supervisado, el aprendizaje no supervisado y el aprendizaje por refuerzo."),
    ("¿Qué es el aprendizaje profundo?", "El aprendizaje profundo es una técnica de aprendizaje automático que utiliza redes neuronales profundas para modelar datos complejos."),
    ("¿Qué es una red neuronal?", "Una red neuronal es un modelo computacional inspirado en el cerebro humano, que se utiliza para reconocer patrones y hacer predicciones."),
    ("¿Qué son los algoritmos de clasificación?", "Los algoritmos de clasificación son métodos que asignan etiquetas a los datos en función de características aprendidas a partir de datos de entrenamiento."),
    ("¿Qué es la regresión en aprendizaje automático?", "La regresión es una técnica que se utiliza para predecir un valor continuo a partir de un conjunto de características."),
    ("¿Qué es el procesamiento de lenguaje natural?", "El procesamiento de lenguaje natural es una rama de la inteligencia artificial que se centra en la interacción entre computadoras y humanos a través del lenguaje natural."),
    ("¿Cómo se aplica la inteligencia artificial en la vida diaria?", "La inteligencia artificial se utiliza en asistentes virtuales, recomendaciones de productos, diagnósticos médicos y más."),
    ("¿Qué son los datos estructurados y no estructurados?", "Los datos estructurados son datos organizados en un formato específico, mientras que los datos no estructurados no siguen un formato predefinido, como texto o imágenes."),
    ("¿Qué es la ética en la inteligencia artificial?", "La ética en la inteligencia artificial se refiere a las consideraciones sobre el uso responsable y justo de la tecnología y su impacto en la sociedad."),
    ("¿Qué son los chatbots?", "Los chatbots son programas diseñados para simular conversaciones con usuarios, utilizando procesamiento de lenguaje natural y aprendizaje automático."),
    ("¿Qué es la visión por computadora?", "La visión por computadora es un campo de la inteligencia artificial que permite a las computadoras interpretar y comprender imágenes y videos."),
    ("¿Cómo se entrenan los modelos de inteligencia artificial?", "Los modelos de inteligencia artificial se entrenan utilizando conjuntos de datos etiquetados y ajustando los parámetros para minimizar el error en las predicciones."),
    ("¿Qué es la sobreajuste en modelos de aprendizaje automático?", "El sobreajuste es un problema que ocurre cuando un modelo se adapta demasiado a los datos de entrenamiento y no generaliza bien a nuevos datos."),
    ("¿Qué es un modelo generativo?", "Un modelo generativo es un tipo de modelo que puede generar nuevos datos similares a los datos de entrenamiento que ha visto."),
    ("¿Qué es la inteligencia artificial general?", "La inteligencia artificial general es un concepto teórico donde una máquina tiene la capacidad de entender, aprender y aplicar inteligencia en una variedad de tareas, similar a la inteligencia humana."),
]

todas_las_listas = {
    "Conceptos Básicos AI": conceptos_basicos_ai,
    "Aprendizaje Automático": aprendizaje_automatico,
    "Redes Neuronales": redes_neuronales,
    "Saludos y Conversaciones": saludos_y_conversaciones,
    "Base de Conocimiento AI": base_conocimiento_ai
}

# Guarda cada tema en una hoja separada en un archivo Excel
with pd.ExcelWriter('preguntas_respuestas_divididas.xlsx') as writer:
    for tema, items in todas_las_listas.items():
        df = pd.DataFrame(items, columns=["Pregunta", "Respuesta"])
        df.to_excel(writer, sheet_name=tema, index=False)
