import re

text_t = """\Mr. Harvey Lights a Candle\"" is anchored by a brilliant performance by Timothy Spall.<br /><br />While 
we can predict that his titular morose, up tight teacher will have some sort of break down or catharsis based on some 
deep down secret from his past, how his emotions are unveiled is surprising. Spall's range of feelings conveyed is 
quite moving and more than he usually gets to portray as part of the Mike Leigh repertory.<br /><br />While an 
expected boring school bus trip has only been used for comic purposes, such as on \""The Simpsons,\"" this central 
situation of a visit to Salisbury Cathedral in Rhidian Brook's script is well-contained and structured for dramatic 
purposes, and is almost formally divided into acts.<br /><br />We're introduced to the urban British range of 
racially and religiously diverse kids (with their uniforms I couldn't tell if this is a \""private\"" or \""public\"" 
school), as they gather  the rapping black kids, the serious South Asians and Muslims, the white bullies and mean 
girls  but conveyed quite naturally and individually. The young actors, some of whom I recognized from British TV 
such as \""Shameless,\"" were exuberant in representing the usual range of junior high social pressures. Celia Imrie 
puts more warmth into the supervisor's role than the martinets she usually has to play.<br /><br />A break in the 
trip leads to a transformative crisis for some while others remain amusingly oblivious. We think, like the teacher 
portrayed by Ben Miles of \""Coupling,\"" that we will be spoon fed a didactic lesson about religious tolerance, 
but it's much more about faith in people as well as God, which is why the BBC showed it in England at Easter time and 
BBC America showed it in the U.S. over Christmas.<br /><br />Nathalie Press, who was also so good in \""Summer of 
Love,\"" has a key role in Mr. Harvey's redemption that could have been played for movie-of-the-week preaching, 
but is touching as they reach out to each other in an unexpected way (unfortunately I saw their intense scene 
interrupted by commercials).<br /><br />While it is a bit heavy-handed in several times pointedly calling this road 
trip \""a pilgrimage,\"" this quiet film was the best evocation of \""good will towards men\"" than I've seen in most 
holiday-themed TV movies."""""

# </?[^>]*>

text_u = re.sub(r"</?[^>]*>|\\|'\w*|[^(\w|\s)]", ' ', text_t)
# text_u = re.sub(r'\\', '', text_u)
# text_u = text_u.lower()
print(text_u)
print(text_t)
print(len(text_u))
print(len(text_t))
