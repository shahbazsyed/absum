import syntok.segmenter as segmenter
from summarizers import textrank

text = """The Charlie Hebdo cartoonists were smarter theologians than the jihadis.

Rather disturbingly, one word seems to connect the activity of the Paris terrorists and that of the Charlie Hebdo cartoonists: iconoclasm. I say disturbingly, because pointing out some common ground may be seen as blurring the crucial distinction between murderous bastards and innocent satirists. 

Nonetheless, it strikes me as fascinating that the cartoonists were profoundly iconoclastic in their constant ridiculing of religion (all religions, it must be noted) – yet it is precisely this same ancient tradition of iconoclasm that inspires Jews and Muslims to resist representational art and, in its most twisted pathological form, to attack the offices of a Paris magazine and slaughter those whose only weapon was the pen. So what’s the connection? 

In one sense an iconoclast is someone who refuses the established view of things, who kicks out against cherished beliefs and institutions. Which sounds pretty much like Charlie Hebdo. But the word iconoclast also describes those religious people who refuse and smash representational images, especially of the divine. The second of the Ten Commandments prohibits graven images – which is why there are no pictures of God in Judaism or Islam. And theologically speaking, the reason they are deeply suspicious of divine representation is because they fear that such representations of God might get confused for the real thing. The danger, they believe, is that we might end up overinvesting in a bad copy, something that looks a lot like what we might think of as god, but which, in reality, is just a human projection. So much better then to smash all representations of the divine. 

And yet this, of course, is exactly what Charlie Hebdo was doing. In the bluntest, rudest, most scatological and offensive of terms, Charlie Hebdo has been insisting that the images people worship are just human creations – bad and dangerous human creations. And in taking the piss out of such images, they actually exist in a tradition of religious iconoclasts going back as far as Abraham taking a hammer to his father’s statues. Both are attacks on representations of the divine. Which is why the terrorists, as well as being murderers, are theologically mistaken in thinking Charlie Hebdo is the enemy. For if God is fundamentally unrepresentable, then any representation of God is necessarily less than God and thus deserves to be fully and fearlessly attacked. And what better way of doing this than through satire, like scribbling a little moustache on a grand statue of God. 
"""

summary = textrank.summarize(text, weight_function='edit_distance')
print(summary)