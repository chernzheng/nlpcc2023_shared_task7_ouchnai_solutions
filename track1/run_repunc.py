from paddlespeech.cli.text.infer import TextExecutor
from difflib import SequenceMatcher
import json

text_punc = TextExecutor()

w = open("/opt/models/repunc_train.json", 'w+')

repunc_result = []
with open("/opt/models/train.json", 'r') as f:
    data = f.readlines()
    essays = json.loads(data[0])
    for essay in essays:
        paras = essay['Text']
        id = essay["Id"]
        Logicgrade = essay["Logicgrade"]
        num_delete_comma = 0
        num_insert_comma = 0
        num_replace_comma_to_period = 0
        num_delete_period = 0
        num_insert_period = 0
        num_replace_period_to_comma = 0
        num_equal_comma = 0
        num_equal_period = 0
        count_comma = 0
        count_period = 0
        for para in paras:
            result = text_punc(text=para)
            # {冒号，顿号} -> 读号
            # {问号, 叹号，分号} -> 句号
            para   =   para.replace('：', '，').replace('、', '，').replace('？', '。').replace('！', '。').replace('；', '。').replace('“', '').replace('”', '').replace(',', '，').replace(':', '，')
            result = result.replace('：', '，').replace('、', '，').replace('？', '。').replace('！', '。').replace('；', '。')
            for i in para:
                if i == '，':
                    count_comma += 1
                elif i == '。':
                    count_period += 1
            matcher = SequenceMatcher(None, para, result)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'delete':
                    if para[i1:i2] == '，':
                        num_delete_comma += 1
                    elif para[i1:i2] == '。':
                        num_delete_period += 1
                    else:
                        print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, para[i1:i2], result[j1:j2]))
                elif tag == 'insert':
                    if result[j1:j2] == '，':
                        num_insert_comma += 1
                    elif result[j1:j2] == '。':
                        num_insert_period += 1
                    else:
                        print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, para[i1:i2], result[j1:j2]))
                elif tag == 'replace':
                    if '，' in para[i1:i2] and '。' in result[j1:j2]:
                        num_replace_comma_to_period += 1
                    elif '。' in para[i1:i2] and '，' in result[j1:j2]:
                        num_replace_period_to_comma += 1
                    else:
                        print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, para[i1:i2], result[j1:j2]))
                elif tag == 'equal':
                    if '，' in para[i1:i2]:
                        num_equal_comma += 1
                    elif '。' in para[i1:i2]:
                        num_equal_period += 1

        repunc_result.append({"Id":id, 
                              "Logicgrade":Logicgrade, 
                              "features": [num_delete_comma,
                                           num_insert_comma,
                                           num_replace_comma_to_period,
                                           num_delete_period,
                                           num_insert_period,
                                           num_replace_period_to_comma,
                                           num_equal_comma,
                                           num_equal_period,
                                           num_equal_comma/count_comma,
                                           num_equal_period/count_period]})            

w.write(json.dumps(repunc_result))
w.close()
