

class BPE():
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.token2id = {}
        self.id2token = {}
        
    def fit(self, text):
        unique_tokens = set(text)
        sorted_tokens = sorted(unique_tokens)
        
        tokens = list(text)
        for i in range(self.vocab_size - len(sorted_tokens)):
            dict_tokens = {}
            # ищем самый частый токен
            for j in range(len(tokens) - 1):
                pair = tokens[j] + tokens[j+1]
                if pair in dict_tokens:
                    dict_tokens[pair] += 1
                else:
                    dict_tokens[pair] = 1
            most_frequent = max(dict_tokens, key=lambda x: dict_tokens[x])
            
            # поиск и замена токена 
            for j in range(len(tokens) - 1):
                pair = tokens[j] + tokens[j+1]
                if pair == most_frequent:
                    tokens[j] = most_frequent
                    tokens[j+1] = ''
            tokens = [x for x in tokens if x != '']
            # складываем найденные занчения в список
            sorted_tokens.append(most_frequent)

        self.token2id = dict(zip(sorted_tokens, range(self.vocab_size)))
        self.id2token = dict(zip(range(self.vocab_size), sorted_tokens))
        
    def encode(self,text):
        # список всех символов
        tokens = list(text)
        # длина списка всех 
        len_tokens = len(tokens)
        # id самого длиного токена
        id_max_len = max(self.id2token, key=lambda x: len(self.id2token[x]))
        # длина самого длинного токена
        token_max_len = len(self.id2token[id_max_len])
        res_seq = []
        k = 0
        i = 0
        while i < len(tokens):
            if k != 0:
                i = i + k
            # первый символ и первое значение в словаре для него
            first_value = self.token2id[tokens[i]]
            first_item = tokens[i]
            
            item = ''
            k = 0
            
            # ищем самый длинный токен 
            for j in range(0, token_max_len + 1):
                # Если приближаемся к концу списка
                if i + j == len_tokens - 1:
                    break
                item = item + tokens[i + j]
                value = self.token2id.get(item, first_value)
                if value != first_value and len(first_item) < len(item):
                    first_value = value
                    first_item = item
                    k = j 
            # складываем найденные занчения в список
            res_seq.append(first_value)
            i = i + 1
        return res_seq
    
    # раскодировать список цифр
    def decode(self, token_ids):
        string = ''
        for token_id in token_ids:
            token = self.id2token[token_id]
            string = string + token
        return string
bpe = BPE(30)
bpe.fit('Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.')
print(bpe.encode('Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'))
