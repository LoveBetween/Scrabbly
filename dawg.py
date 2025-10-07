class DawgNode:
    NextId = 0

    def __init__(self):
        self.id = DawgNode.NextId
        DawgNode.NextId += 1
        self.final = False
        self.edges = {}

        self.count = 0

    def num_reachable(self):
        if self.count:
            return self.count

        count = 0
        if self.final:
            count += 1
        for node in self.edges.values():
            count += node.num_reachable()
        self.count = count
        return count


class Dawg:
    def __init__(self):
        self.previous_word = ""
        self.root = DawgNode()

        self.unchecked_nodes = []
        self.minimized_nodes = {}
        self.data = []

    def insert(self, word, data):
        if word <= self.previous_word:
            print("Error: Words must be inserted in alphabetical order.")

        common_prefix = 0
        min_length = min(len(word), len(self.previous_word))
        for i in range(min_length):
            if word[i] != self.previous_word[i]:
                break
            common_prefix += 1

        self.minimize(common_prefix)

        self.data.append(data)

        if not self.unchecked_nodes:
            node = self.root
        else:
            node = self.unchecked_nodes[-1][2]
        for letter in word[common_prefix:]:
            next_node = DawgNode()

            node.edges[letter] = next_node
            self.unchecked_nodes.append((node, letter, next_node))
            node = next_node
        node.final = True
        self.previous_word = word

    def finish(self):
        self.minimize(0)
        self.root.num_reachable()

    def minimize(self, down_to):
        for i in range(len(self.unchecked_nodes) - 1, down_to - 1, -1):
            parent, letter, child = self.unchecked_nodes[i]
            if child in self.minimized_nodes:
                parent.edges[letter] = self.minimized_nodes[child]
            else:
                self.minimized_nodes[child] = child
            self.unchecked_nodes.pop()

    def node_count(self):
        return len(self.minimized_nodes)

    def edge_count(self):
        count = 0
        for node in self.minimized_nodes.values():
            count += len(node.edges)
        return count

    def lookup(self, word):
        node = self.root
        skipped = 0
        for letter in word:
            if letter not in node.edges:
                for label, child in sorted(node.edges.items()):
                    if label == letter:
                        if node.final:
                            skipped += 1
                            node = child
                            break
                        skipped += child.count
            else:
                node = node.edges[letter]
        if node.final:
            return self.data[skipped]

    def find_words_rec(self, node, letters, word, words):
        if node.final:
            words.add(word)
        for label, child in node.edges.items():
            for i, letter in enumerate(letters):
                if label == letter or letter == "*":
                    new_letters = letters[:i] + letters[i+1:]
                    self.find_words_rec(child, new_letters, word + label, words)
        return words

    def find_words(self, letters):
        words = set()
        return self.find_words_rec(self.root, letters, "", words)

    def setup(self, words):
        #start_time = time.time()

        count = 0
        for word in words:
            self.insert(word, count)
            count += 1
        self.finish()

        #end_time = time.time()
        #"print(f"Dictionary loaded in {end_time - start_time:.2f}s")

        print(f"nodecount : {self.node_count()}")
        print(f"edgecount : {self.edge_count()}")
    def check_valid_word(self, word):
        node = self.root
        for letter in word:
            if letter not in node.edges:
                return False
            node = node.edges[letter]
        return node.final

    def check_placement(self, placement, letters):
        def check_placement_rec(placement, node, letters, new_word, words, is_connected, letter_placed):
            if node.final and is_connected and letter_placed and (len(placement['word']) <= len(new_word) and placement['next_letter'] == "_" or placement['word'][len(new_word)] == "_"):
                words.add(new_word)
            if len(new_word) == len(placement['word']):
                return
            next_letter = placement['word'][len(new_word)]
            if next_letter != "_":
                if next_letter in node.edges:
                    child = node.edges[next_letter]
                    check_placement_rec(placement, child, letters, new_word + next_letter, words, True, letter_placed)
            else:
                for i, letter in enumerate(letters):
                    if letter in node.edges or letter == "*":
                        if len(placement['perpendicular'][len(new_word)]) < 2 or self.check_valid_word(placement['perpendicular'][len(new_word)].replace("_", letter)):
                            if len(placement['perpendicular'][len(new_word)]) > 1:
                                is_connected = True
                            new_letters = letters[:i] + letters[i + 1:]
                            child = node.edges[letter]
                            check_placement_rec(placement, child, new_letters, new_word + letter, words, is_connected, True)
            return words

        words = set()
        return check_placement_rec(placement, self.root, letters, "", words, False, False)


