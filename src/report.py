import numpy as np

from pythainlp import word_tokenize
from keras.models import load_model

np.set_printoptions(4)

def get_vocab_wvs(wv_path, preprocessed_doc=None, vocabs=None):
    fasttext_fp = open(wv_path, encoding='utf-8-sig')
    white_spaces = ['', ' ']
    
    if(not(vocabs) and preprocessed_doc):
        vocabs = set([tk for tk in preprocessed_doc if tk not in white_spaces])

    vocab_wvs = {}

    line_count = 0
    vocab_count = 0
    for line in fasttext_fp:
        if(line_count > 0):
            line = line.split()
            if(vocab_count < len(vocabs)):
                if(line[0] in vocabs):
                    vocab_wvs[line[0]] = line[1:]
                    print('found %s %s total_len: %s' % (line_count, line[0], len(vocabs)))
                    vocab_count += 1
                    print(vocab_count)
            else:
                break
        line_count += 1
    
    return vocab_wvs

def vectorize_tokens(sentence, vocab_wvs=None, wvl=300):
    word_vectors = np.zeros((len(sentence), wvl))
    for i in range(len(sentence)):
        try:
            if(sentence[i] != '<PAD>'):
                word_vectors[i, :] = vocab_wvs[sentence[i]]
        except:
            pass

    return word_vectors

if __name__ == "__main__":
    s1 = 'ศาสนาพุทธเป็นศาสนาอเทวนิยมปฏิเสธการมีอยู่ของพระเป็นเจ้าหรือพระผู้สร้างและเชื่อในศักยภาพของมนุษย์ว่าทุกคนสามารถพัฒนาจิตใจไปสู่ความเป็นมนุษย์ที่สมบูรณ์ได้ด้วยความเพียรของตนกล่าวคือศาสนาพุทธสอนให้มนุษย์บันดาลชีวิตของตนเองด้วยผลแห่งการกระทำของตนตามกฎแห่งกรรม'
    s2 = 'มีพระธรรมที่พระองค์ตรัสรู้ชอบด้วยพระองค์เองและตรัสสอนไว้เป็นหลักคำสอนสำคัญมีพระสงฆ์สาวกผู้ตัดสินใจออกบวชเพื่อศึกษาปฏิบัติตนตามคำสั่งสอนธรรมวินัยของพระบรมศาสดาเพื่อบรรลุสู่จุดหมายคือพระนิพพานและสร้างสังฆะเป็นชุมชนเพื่อสืบทอดคำสอนของพระบรมศาสดารวมเรียกว่าพระรัตนตรัย'
    s3 = 'พระพุทธเจ้าพระองค์ปัจจุบันคือพระโคตมพุทธเจ้ามีพระนามเดิมว่าเจ้าชายสิทธัตถะได้ทรงเริ่มออกเผยแผ่คำสอนในชมพูทวีปตั้งแต่สมัยพุทธกาล'
    s4 = 'อัลเบิร์ต ไอน์สไตน์เป็นศาสตราจารย์ทางฟิสิกส์และนักฟิสิกส์ทฤษฎีชาวเยอรมันเชื้อสายยิวซึ่งเป็นที่ยอมรับกันอย่างกว้างขวางว่าเป็นนักวิทยาศาสตร์ที่ยิ่งใหญ่ที่สุดในคริสต์ศตวรรษที่20เขาเป็นผู้เสนอทฤษฎีสัมพัทธภาพและมีส่วนร่วมในการพัฒนากลศาสตร์ควอนตัมกลศาสตร์สถิติและจักรวาลวิทยา'
    s5 = 'นิโคลาเทสลาเป็นนักประดิษฐ์นักฟิสิกส์วิศวกรเครื่องกลวิศวกรไฟฟ้าาวเซอร์เบียอเมริกันเขาเกิดที่ Smiljan ในอดีตออสเตรียฮังการีซึ่งปัจจุบันคือสาธารณรัฐโครเอเชียเทสลาเป็นผู้คิดค้นสัญญาณวิทยุการค้นพบหลักการสนามแม่เหล็กไฟฟ้าและผลงานที่ทำให้เขาเป็นที่รู้จักกันดีคือการค้นคว้าพัฒนาไฟฟ้ากระแสสลับ'
    s6 = 'วัตถุบินกำหนดเอกลักษณ์ไม่ได้หรือมักเรียกว่ายูเอฟโอหรือยูโฟ UFO ในความหมายกว้างที่สุดคือสิ่งผิดปกติประจักษ์ชัดใดๆในท้องฟ้าซึ่งไม่สามารถระบุเอกลักษณ์ได้ในทันทีว่าเป็นวัตถุหรือปรากฏการณ์ใดๆที่ทราบจากการสังเกตด้วยตาหรือการใช้เครื่องมือช่วยเช่นเรดาร์สิ่งผิดปกตินี้มักเรียกว่าจานผีหรือจานบิน'
    s7 = 'รถสูตรหนึ่งหรือฟอร์มูลาวันหรือเอฟวันเป็นการแข่งขันรถระดับสูงสุดจากความช่วยเหลือของสมาพันธ์รถยนต์นานาชาติคำว่าสูตรหมายถึงกฎกติกาที่ผู้เข้าแข่งขันและรถทุกคันต้องปฏิบัติตามฤดูกาลแข่งขันของเอฟวันประกอบด้วยการแข่งขันหลายครั้งหรือที่เรียกว่ากรังด์ปรีซ์ตามวัตถุประสงค์การสร้างของสนามแข่ง'
    s8 = 'โลมาเป็นสัตว์เลี้ยงลูกด้วยน้ำนมจำพวกหนึ่งอาศัยอยู่ทั้งในทะเลน้ำจืดและน้ำกร่อยมีรูปร่างคล้ายปลาคือมีครีบมีหางแต่โลมามิใช่ปลาเพราะเป็นสัตว์เลี้ยงลูกด้วยน้ำนมที่มีรกจัดอยู่ในอันดับวาฬและโลมาซึ่งประกอบไปด้วยวาฬและโลมาซึ่งโลมาจะมีขนาดเล็กกว่าวาฬมากและจัดอยู่ในกลุ่มวาฬมีฟันเท่านั้น'
    s9 = 'งูเห่าเป็นงูพิษขนาดกลางที่อยู่ในสกุล Naja ในวงศ์งูพิษเขี้ยวหน้า Elapidae วงศ์ย่อย Elapinae ซึ่งเป็นสกุลของงูพิษที่อาจเรียกได้ว่าเป็นที่รู้จักกันดีที่สุดอย่างหนึ่งเมื่อตกใจหรือต้องการขู่ศัตรูมักทำเสียงขู่ฟู่และแผ่แผ่นหนังที่อยู่หลังบริเวณคอออกเป็นแผ่นด้านข้างเรียกว่าแม่เบี้ย'

    s1_tkned = word_tokenize(s1)
    s2_tkned = word_tokenize(s2)
    s3_tkned = word_tokenize(s3)
    s4_tkned = word_tokenize(s4)
    s5_tkned = word_tokenize(s5)
    s6_tkned = word_tokenize(s6)
    s7_tkned = word_tokenize(s7)
    s8_tkned = word_tokenize(s8)
    s9_tkned = word_tokenize(s9)

    sentences = []
    sentences.append(s1_tkned)
    sentences.append(s2_tkned)
    sentences.append(s3_tkned)
    sentences.append(s4_tkned)
    sentences.append(s5_tkned)
    sentences.append(s6_tkned)
    sentences.append(s7_tkned)
    sentences.append(s8_tkned)
    sentences.append(s9_tkned)

    vocabs = set([tk for s in sentences for tk in s if not tk in ['', ' ']])
    print(len(vocabs))

    vocab_wvs = get_vocab_wvs('C:/Users/Patdanai/Workspace/cc.th.300.vec', vocabs=vocabs)
    print(len(vocab_wvs))
    
    for i in range(len(sentences)):
        while(len(sentences[i]) < 40):
            sentences[i].insert(0, '<PAD>')
        while(len(sentences[i]) > 40):
            sentences[i].pop()

    es = np.zeros((9, 40, 300))
    for i in range(len(es)):
        es[i] = vectorize_tokens(sentences[i], vocab_wvs=vocab_wvs)

    model = load_model('./model/lstm_compare_model_2019_05_03_18_41_55.879999_2048.h5')

    similarity_matrix = np.zeros((es.shape[0], es.shape[0]))
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            if(i == j):
                similarity_matrix[i][j] = -1.
            else:
                similarity_matrix[i][j] = model.predict([np.expand_dims(es[i], axis=0), np.expand_dims(es[j], axis=0)])

    print(similarity_matrix)
    np.savetxt('report.txt', similarity_matrix, fmt='%.4f')

