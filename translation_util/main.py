from PyQt5 import QtWidgets
import translate as form
import json
import sys
import yaml
import codecs
import os


class WindowAppText(QtWidgets.QMainWindow, form.Ui_MainWindow):

    def __init__(self, config_path):
        super().__init__()

        self.config_path = config_path

        with codecs.open(config_path, 'r', 'utf-8') as fin:
            self.config = yaml.load(fin)

        with codecs.open(self.config['data_path'], 'r', 'utf-8') as fin:
            self.data = json.load(fin)

        self.setupUi(self)
        self.current_id = self.config['last_id']
        self.init_windows()

        self.next_sample.clicked.connect(self.next)
        self.save_translate.clicked.connect(self.save)
        self.colored_conversation.clicked.connect(self.set_colored)
        self.copy_conversation.clicked.connect(self.conversation_to_buffer)
        self.copy_text.clicked.connect(self.text_to_buffer)

    def conversation_to_buffer(self):
        self.conversation.selectAll()
        self.conversation.copy()

    def text_to_buffer(self):
        self.text.selectAll()
        self.text.copy()

    def save_sample(self, sample):
        try:
            save_path = self.config['data_path']
            save_path = save_path.replace('CoQA', 'CoQA_ru')
            if os.path.exists(save_path):
                with codecs.open(save_path, 'r', 'utf-8') as fin:
                    translated_data = json.load(fin)
            else:
                translated_data = []
            translated_data.append(sample)
            with codecs.open(save_path, 'w', 'utf-8') as fout:
                json.dump(translated_data, fout)
        except Exception as e:
            raise IOError('сохранение сэмпла')

    def save_config(self):
        try:
            self.config['last_id'] = self.current_id + 1
            with codecs.open(self.config_path, 'w', 'utf-8') as fout:
                yaml.dump(self.config, fout)
        except:
            raise IOError('обновления конфигурации')

    def update_conversation(self, sample):
        text = self.conversation_translate.toPlainText()
        if len(text) < 1:
            raise IOError('нет перевода беседы')
        text = text.split('\n')
        for idx, (q, a, r) in enumerate(zip(text[0::3], text[1::3], text[2::3])):
            sample['questions'][idx]['input_text'] = q[q.find('-')+1:].strip()
            sample['answers'][idx]['input_text'] = a[a.find('-')+1:].strip()
            sample['answers'][idx]['span_text'] = r.strip()[1:-1]
        return sample

    def set_colored(self):
        text = self.conversation_translate.toPlainText()
        if text:
            sample = self.update_conversation(self.data[self.current_id])
            self.conversation_translate.clear()
            self.conversation_translate.appendHtml(self.get_conversation(sample))

    def save(self):
        try:
            sample = self.data[self.current_id]
            text = self.text_translate.toPlainText()
            if len(text) < 1:
                raise IOError('нет перевода текста')
            sample = self.update_conversation(sample)
            sample['story'] = text
            sample['checked'] = self.checkBox.isChecked()
            self.save_sample(sample)
            self.save_config()
        except IOError as e:
            self.status.setText(f"Статус фиксации: ошибка {e}")
        else:
            self.status.setText("Статус фиксации: сохранено")
            self.save_translate.setDisabled(True)
            self.next_sample.setDisabled(False)

    def next(self):
        self.current_id += 1
        self.init_windows()

    def init_windows(self):
        self.current.setText(f"Обработано: {self.current_id}")
        self.full.setText(f"Осталось {len(self.data) - self.current_id}")
        self.text.setText(self.data[self.current_id]['story'])
        self.conversation.setText(self.get_conversation(self.data[self.current_id]))
        self.status.setText("Статус фиксации: не зафиксировано")
        self.save_translate.setDisabled(False)
        self.next_sample.setDisabled(True)
        self.checkBox.setChecked(False)
        self.conversation_translate.clear()
        self.text_translate.clear()

    def get_conversation(self, sample):
        blue_start = "<span style=\"color:blue;\" >"
        green_start = "<span style=\"color:green;\" >"
        span_end = "</span>"
        conversation = ''
        for idx, (q, a) in enumerate(zip(sample['questions'], sample['answers']), 1):
            conversation += f"{blue_start}({idx}) - {q['input_text']}{span_end}{'<br>'}"
            conversation += f"{green_start}({idx}) - {a['input_text']}{span_end}{'<br>'}"
            conversation += f"({a['span_text']}){'<br>'}"
        return conversation

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    config_path = 'config.yml'
    window = WindowAppText(config_path)
    window.show()
    sys.exit(app.exec_())