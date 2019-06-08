import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(subject='iPython'):
    # 设置smtplib所需的参数
    # 下面的发件人，收件人是用于邮件传输的。
    username = 'ultrainren@163.com'
    password = '163Be1Man'
    sender = 'ultrainren@163.com'
    # receiver='XXX@126.com'
    # 收件人为多个收件人
    receiver = ['ultrain@126.com', 'ultrainren@163.com']

    # 通过Header对象编码的文本，包含utf-8编码信息和Base64编码信息。以下中文名测试ok
    # subject = '中文标题'
    # subject=Header(subject, 'utf-8').encode()

    # 构造邮件对象MIMEMultipart对象
    # 下面的主题，发件人，收件人，日期是显示在邮件页面上的。
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = 'ultrainren@163.com <ultrainren@163.com>'

    # 收件人为多个收件人,通过join将列表转换为以;为间隔的字符串
    msg['To'] = ";".join(receiver)
    # msg['Date']='2012-3-16'

    # 构造文字内容
    text = "Hi!\n...\nBye!\n----DCRA Mark1, Robotlab-XPS"
    text_plain = MIMEText(text, 'plain', 'utf-8')
    msg.attach(text_plain)

    # 发送邮件
    smtp = smtplib.SMTP()
    smtp.connect('smtp.163.com')
    # 我们用set_debuglevel(1)就可以打印出和SMTP服务器交互的所有信息。
    # smtp.set_debuglevel(1)
    smtp.login(username, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()
    print(subject, ' 已发送。')
