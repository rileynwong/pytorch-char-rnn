# PyTorch Char-RNN
PyTorch implementation of a character-level recurrent neural network. See [accompanying blog post](https://www.rileynwong.com/blog/2019/4/24/implementing-char-rnn-from-scratch-in-pytorch-and-generating-fake-book-titles).

Includes pretrained models for generating:
- fake book titles in different genres
- first names in different languages
- constellation names in English and Latin

## Examples
### Book titles
Possible categories in the pretrained model include: `Adult_Fiction, Erotica, Mystery, Romance, Autobiography, Fantasy, New_Adult, Science_Fiction, Biography, Fiction, Nonfiction, Sequential_Art, Childrens, Historical, Novels, Short_Stories, Christian_Fiction, History, Paranormal, Thriller, Classics, Horror, Philosophy, Young_Adult, Contemporary, Humor, Poetry, Dark, Lgbt, Religion`

```
Heart in the Dark (Romance)

Book of the Dark (Fantasy) 

Bed Store (Young Adult)

Growing Me (New Adult)
Me the Bean (New Adult) 

King of the Dark (Erotica)
Your Mind (Erotica)

Red Story (Mystery) 

Be the Life (Biography)
```

### First names
Possible categories in the pretrained model include: `Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Irish, Italian, Japanese, Korean, Polish, Portuguese, Russian, Scottish, Spanish, Vietnamese`

Russian:
```
Rovakov
Uanten
Shantovov
```

Chinese:
```
Chan
Hang
Iun
```

### Constellations
English:
```
orogane
quale
rowans
serpent
kelescop
```

Latin:
```
bearis
corac
serpens
xer
zeriscase
```

## File Overview
- `generate_books.py`: Train/generate fake book titles.
- `generate_names.py`: Train/generate fake first names. 
- `generate_constellations.py`: Train/generate fake constellation names.
- `csv_to_txt.ipynb`: Notebook to convert 
- `models/`: Saved models.
- `data/`: Training data sets as text files.

## Setup
All you really need is [PyTorch](https://pytorch.org/get-started/locally/).

## Usage
At the bottom of each script, the function call `samples(CATEGORY, START_LETTERS)` can be modified for whatever category/starting letters you want. 

`$ python generate_names.py`:
- `samples('Russian', 'RUS)`

`$ python generate_books.py`:
- `samples('Fiction', 'abcdefghijklmnopqrstuvwxyz')`

`$ python generate_constellations.py`
- `samples('latin', 'abcdefghijklmnopqrstuvwxyz')`


### Training your own
Set line 174 and 175 to:
```
rnn = RNN(n_letters, 128, n_letters)
# rnn = torch.load('models/book_titles.pt')
```

Uncomment out this part in the script:
```
print('Training model...')
for iter in range(1, n_iters + 1):
    output, loss = train(*random_training_example())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

### Save model after training
torch.save(rnn, 'language_names.pt')
```

You can also set the training data directory on line 31:
`data_glob = 'data/names/*.txt'` to `data_glob = 'data/your_dir/*.txt'`

and setting each category of text within its own `.txt` file within the `data/your_dir/` directory. 

### Generating your own
If you just want to generate samples, you can comment out the training snippet and set the network to load the pretrained model. i.e. Set line 174 and 175 to:
```
# rnn = RNN(n_letters, 128, n_letters)
rnn = torch.load('models/book_titles.pt')
```
and re-run the script. 


## Credits
- [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [Goodreads data scrape](https://www.kaggle.com/brosen255/goodreads-books)
