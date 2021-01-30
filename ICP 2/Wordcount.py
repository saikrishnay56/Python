fname = ("C:\\Users\\anil7\\OneDrive\\Desktop" "r")

test = open("test.txt", "r")
d = dict()
for line in test:
        line = line.strip()

        words = line.split(" ")
        for word in words:
                if word in d:
                        d[word] = d[word] + 1
                else:
                        d[word] = 1
for key in list(d.keys()):
        print(key, ":", d[key])

f = open("test.txt", "a+")
for key in list(d.keys()):
        f.write("\n")
        f.write(key)
        f.write(" : ")
        f.write(str(d[key]))
f.close()

