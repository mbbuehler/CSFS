


class ERParser:

    @staticmethod
    def parse(token):
        print(token)


def test():
    token = '0:13,1:14,2:1,3:3,4:4,5:7,6:11,7:5,8:12,9:15,10:6,11:2,12:8,13:9,14:10|e7cf0fccca7858d47a96c82837e6d439'
    print(ERParser.parse(token))

if __name__ == '__main__':
    test()