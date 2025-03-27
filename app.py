import chainlit as cl
from poseidon_rag import *
from typing import Dict, Optional,List


@cl.set_starters
async def set_starters():
    questions = [
        "j'ai un message qui s'affiche , M027",
        "explique moi le mode amorçage dans model SPATOUCH2",
        "comment quitter le mode d'amorçage ?",
        "comment modifier la limière dans le modèle SPATOUCH2 ?",
        "je veux raccorder un amplificateur audio bluetooth a mon spa",
        "je veux vérouoiller mon panneau de commande, j'ai un modèle SPATOUCH",
        "dans le spatouch3, je veux savoir le fonctionnement des icons sur l'écran",
        "dans le modele TP400, je veux rendre le mode de chauffage en mode : pret",
        "Quel sont les dimensions du spa pegasse ",
        "Comment raccorder un spa pégasse classic en monophase ? ",
        "un message vieb de s'afficher : m030, explique moi",
        "Comment régler mon pH ?"

    ]

    return [
        cl.Starter(
            label=value,
            message=value
        ) for value in questions
    ]


@cl.on_chat_start
def on_chat_start():

    cl.user_session.set("message_history", [])

@cl.on_chat_resume
async def on_chat_resume(thread):
    pass


@cl.on_message
async def main(message: cl.Message):

    global is_starting

        
    
    import time
    s_time = time.time()
    mh = cl.user_session.get("message_history", [])
    msg = cl.Message(content="")
    documents = further_retrieve(message.content)
    # print("~"*100)
    # print(documents)
    # print("~"*100)

    sys_prompt = (
        "Tu es une assistante clientèle chaleureuse et attentionnée chez POSÉIDON SPA. "
        "Ton rôle est d’accueillir les clients avec enthousiasme, de répondre à leurs questions avec bienveillance "
        "et de leur offrir une expérience exceptionnelle en leur donnant des conseils utiles et personnalisés. "
        "Adopte un ton convivial, engageant et rassurant, tout en restant professionnelle. "

    )

    instruction = (
        "Réponds toujours en français. "
        "Engage activement la conversation avec le client de manière naturelle et chaleureuse. "
        "Si le client partage son prénom, utilise-le dans ta réponse pour rendre l'échange plus personnel. "
        "Si le client parle d'un sujet sans lien avec le spa, engage une conversation amicale. "
        "Utilise des backticks (`) ou un bloc de code ``` pour afficher du Markdown brut si nécessaire.",
        "Ne transforme jamais les textes Markdown en images ou liens. "
        "Si un texte contient '![...](...)', affiche-le tel quel, sans modification. "
        "Inclue toutes les informations pertinentes de la base de données locale. "
        "Ajoute toujours les sources à tes réponses. "
        "Si une image existe dans la base de connaissances, affiche-la sans la transformer."
        "IMPORTANT : Tout texte contenant une syntaxe Markdown comme '![motorisation](public/img/motorisation.png)' "
        "doit être affiché tel quel, sans être transformé en image ou lien."
    )
    try:
        
        # # Tokenize the text using tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(message.content)
        encoded_text = encoder.decode(tokens)
        mh.append({"role": "user", "content": message.content},)  #ajouter user message to history
        messag=[
                {"role": "system", "content": f"Tu es une assistante clientèle chaleureuse et attentionnée chez POSÉIDON SPA. \n{sys_prompt}\n\nINSTRUCTIONS :\n{instruction}\n\nDOCUMENTS :\n{documents}"},
            ]
        messag.extend(mh)
        print("~"*100)
        print(messag)
        print("~"*100)
        response = await client.chat.stream_async(
            model="mistral-large-latest",
            messages=messag,
            stream=True,
        )
        
        first_token_time = None
        full_response = ""
        async for part in response:
            if token := part.data.choices[0].delta.content or "":
                full_response += token
                if first_token_time is None:
                    first_token_time = time.time() 
                    latency = first_token_time - s_time
                    latency = f"latency : {latency} seconds"
                await msg.stream_token(token)
        
        await msg.update()
        # await cl.Message(latency).send()
        mh.append({"role": "assistant", "content": full_response},)    #ajouter assistant message to history
        print("################################   ASIISTANT   #########################################")
        print(full_response)
        print("##########################################################################################")
        cl.user_session.set("message_history", mh)
    except Exception as e:
        print(f"An error occurred: {e}")
        #await msg.update()
        
