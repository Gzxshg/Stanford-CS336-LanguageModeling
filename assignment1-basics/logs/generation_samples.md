# Generation samples (assignment problem: decoding)

Checkpoint: checkpoints/ckpt_best.pt (TinyStories, iteration 9999, best valid 1.3163)
Prompt: "Once upon a time, there was a little girl named Lily."

---

## 1. Greedy (temperature=0)

    uv run generate.py --prompt "Once upon a time, there was a little girl named Lily." --temperature 0 --max-new-tokens 200

Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat yummy food. One day, she found a big, red apple in the kitchen. She was very happy and wanted to eat it all by herself.
Lily took the apple to her room and started to eat it. But then, she heard a loud noise outside. She looked out the window and saw a big, scary dog. The dog wanted the apple too. Lily was scared, but she had an idea.
Lily went outside and found a big, scary dog. She thought the dog would not take her apple. But the dog saw the apple and started to eat it. Lily was very sad and scared. She ran back to her room and cried. The dog ate her apple and left her alone.

Note: degenerates into repetition ("a big, scary dog" twice almost verbatim), abrupt ending.

---

## 2. temperature=0.8, top-p=0.9 (best)

    uv run generate.py --prompt "Once upon a time, there was a little girl named Lily." --temperature 0.8 --top-p 0.9 --max-new-tokens 200

Once upon a time, there was a little girl named Lily. She loved to play outside with her friends. One day, Lily and her friends were playing near a big tree. They were very happy.
Suddenly, they heard a loud sound. It was thunder! They felt scared. Lily's friends hid behind the tree. They were not scared anymore. They all wanted to find a safe place to hide.
The thunder came closer, but Lily and her friends stayed close. They were very quiet. They listened to the thunder. Soon, the thunder went away. Lily's friends came out and were happy. They all said thank you to the thunder. And they played happily ever after.

Note: complete coherent story arc, natural TinyStories style.

---

## 3. temperature=1.2, no top-p (too hot)

    uv run generate.py --prompt "Once upon a time, there was a little girl named Lily." --temperature 1.2 --top-p 1.0 --max-new-tokens 200

Once upon a time, there was a little girl named Lily. She lived in a small house with her mommy, daddy, and an ancient carpet. Lily loved her carpet very much, but her room was always so messy! One day, her room became black, and Lily was sad.
Lily didn't know the carpet belonged to. She wanted it to be her new favorite toy. Her mommy saw and said, "Lily, you need to clean your room now." She washed the carpet, the carpet, and the hands. The carpet became black, and the toys did not fit.
The moral of the story is to always clean up your mess, so you don't have to or anything you don't have to do to fix your things in other responsible tasks.

Note: grammar and semantics break down; low-probability tokens leak in.

---

## 4. OWT model (checkpoints_owt/ckpt_best.pt, best valid 4.6941)

    uv run generate.py --ckpt checkpoints_owt/ckpt_best.pt --tokenizer data/owt_tokenizer.pt --vocab-size 32000 --prompt "The history of the internet began" --temperature 0.8 --top-p 0.9 --max-new-tokens 200

The history of the internet began in 2001, and the industry adopted in a perfect way.

In 2005, a 4.6 billion euro ($7.7 billion) was estimated by 21 percent in 2010 by a third, as a percentage of the income. In 2007, the Asian-American government withdrew a 37 percent increase in 2007, and the United States was 31 percent higher.

The IMF’s GDP rise by 11 percent, and the average annual rate by 2.9 per cent.

The state has not found a slightly more than 60 percent growth since 2009.

Economic leaders have suggested that the current trend increased by 20 percent, while inflation per capita and unemployment rate rose by 1.9 per cent.

“The deficit in unemployment rate has fallen sharply worse. But we’re going to reduce the rising and hopefully we can’t make it more affordable than we are going to,” said Bargel, a federal analyst at the University of Pennsylvania, who has been working

Note: fluent news-style register (numbers, quotes, institutions) but fabricated and
self-contradictory content - expected from a 45M model trained on ~12% of one epoch.
