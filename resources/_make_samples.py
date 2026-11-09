"""Generate modern office fixtures with fictional data."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H = 1240, 1754
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"


def fonts(size: int) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    return ImageFont.truetype(FONT, size), ImageFont.truetype(FONT_B, size)


def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, W - 40, H - 40), outline=(180, 180, 180), width=1)
    return img, draw


def save(img: Image.Image, name: str) -> None:
    path = OUT / name
    img.save(path, quality=92)
    print(path.name, path.stat().st_size)


def make_zayavlenie() -> None:
    img, draw = new_page()
    regular, bold = fonts(28)
    small, _ = fonts(24)
    tiny, _ = fonts(20)

    draw.text((700, 80), "Директору ООО «СеверТорг»", font=regular, fill="black")
    draw.text((700, 120), "Иванову И.И.", font=regular, fill="black")
    draw.text((700, 180), "от менеджера отдела продаж", font=regular, fill="black")
    draw.text((700, 220), "Петровой Анны Сергеевны", font=regular, fill="black")
    draw.text((520, 360), "ЗАЯВЛЕНИЕ", font=bold, fill="black")

    body = [
        "Прошу предоставить мне ежегодный оплачиваемый отпуск",
        "с 15.07.2025 по 28.07.2025 включительно (14 календарных дней)",
        "согласно графику отпусков на 2025 год.",
        "",
        "Дополнительных оснований не имею.",
    ]
    y = 460
    for line in body:
        draw.text((100, y), line, font=regular, fill="black")
        y += 48

    draw.text((100, 780), "15.06.2025", font=regular, fill="black")
    draw.text((780, 780), "______________ / Петрова А.С. /", font=regular, fill="black")
    draw.text((100, 860), "Согласовано:", font=small, fill="black")
    draw.text(
        (100, 910),
        "Начальник отдела ________________ / Сидоров П.П. /",
        font=small,
        fill="black",
    )
    draw.text(
        (100, 980),
        "HR-отдел, таб. № 1042, Contract ID: HR-2025-078",
        font=tiny,
        fill=(60, 60, 60),
    )
    save(img, "zayavlenie_otpusk.png")


def make_spravka() -> None:
    img, draw = new_page()
    regular, bold = fonts(28)
    small, _ = fonts(24)

    draw.text((100, 80), "ООО «СеверТорг»", font=bold, fill="black")
    draw.text(
        (100, 125),
        "ИНН 7701234567  КПП 770101001  ОГРН 1027700123456",
        font=small,
        fill="black",
    )
    draw.text(
        (100, 165),
        "125009, г. Москва, ул. Тверская, д. 10, офис 301",
        font=small,
        fill="black",
    )
    draw.text(
        (100, 205),
        "тел. +7 (495) 123-45-67  e-mail: hr@severtorg.example",
        font=small,
        fill="black",
    )
    draw.line((100, 250, W - 100, 250), fill=(0, 0, 0), width=2)
    draw.text((480, 300), "СПРАВКА", font=bold, fill="black")
    draw.text((330, 350), "№ 145-к от 09.11.2025", font=regular, fill="black")

    lines = [
        "Дана Петровой Анне Сергеевне в том, что она действительно",
        "работает в ООО «СеверТорг» в должности менеджера отдела продаж",
        "с 01.03.2022 по настоящее время.",
        "",
        "Средняя заработная плата за последние 3 месяца составляет",
        "85 000 (восемьдесят пять тысяч) рублей 00 копеек.",
        "",
        "Справка выдана по месту требования.",
    ]
    y = 430
    for line in lines:
        draw.text((100, y), line, font=regular, fill="black")
        y += 48

    draw.text((100, 900), "Генеральный директор", font=regular, fill="black")
    draw.text((700, 900), "______________ / Иванов И.И. /", font=regular, fill="black")
    draw.text((100, 980), "Главный бухгалтер", font=regular, fill="black")
    draw.text((700, 980), "______________ / Козлова М.А. /", font=regular, fill="black")
    draw.text((100, 1100), "М.П.", font=small, fill="black")
    draw.ellipse((140, 1140, 320, 1320), outline=(180, 40, 40), width=3)
    draw.text((175, 1210), "ООО", font=small, fill=(180, 40, 40))
    draw.text((155, 1245), "СеверТорг", font=small, fill=(180, 40, 40))
    save(img, "spravka_s_mesta_raboty.png")


def make_nakladnaya() -> None:
    img, draw = new_page()
    regular, bold = fonts(26)
    small, _ = fonts(22)
    tiny, _ = fonts(20)

    draw.text(
        (100, 70),
        "Универсальный передаточный документ (фрагмент)",
        font=small,
        fill="black",
    )
    draw.text(
        (100, 110),
        "ТОВАРНАЯ НАКЛАДНАЯ № 458 от 03.11.2025",
        font=bold,
        fill="black",
    )
    draw.text((100, 160), "Поставщик: ООО «СеверТорг», г. Москва", font=regular, fill="black")
    draw.text(
        (100, 200),
        "Покупатель: ООО «Ромашка», г. Санкт-Петербург",
        font=regular,
        fill="black",
    )
    draw.text(
        (100, 240),
        "Основание: Договор поставки № 88 от 12.01.2025",
        font=regular,
        fill="black",
    )

    headers = [
        (100, "№"),
        (160, "Наименование"),
        (620, "Ед."),
        (720, "Кол-во"),
        (860, "Цена"),
        (1020, "Сумма"),
    ]
    y0 = 310
    draw.rectangle((90, y0, 1180, y0 + 50), outline="black", width=1)
    for x, text in headers:
        draw.text((x, y0 + 12), text, font=small, fill="black")

    rows = [
        ("1", "Бумага А4, пачка 500 л.", "шт", "20", "320,00", "6 400,00"),
        ("2", "Картридж HP 83A", "шт", "2", "4 900,00", "9 800,00"),
        ("3", "Степлер металлический", "шт", "5", "450,00", "2 250,00"),
    ]
    y = y0 + 50
    for row in rows:
        draw.rectangle((90, y, 1180, y + 48), outline="black", width=1)
        vals = [
            (100, row[0]),
            (160, row[1]),
            (620, row[2]),
            (720, row[3]),
            (860, row[4]),
            (1000, row[5]),
        ]
        for x, text in vals:
            draw.text((x, y + 12), text, font=tiny, fill="black")
        y += 48

    draw.rectangle((90, y, 1180, y + 48), outline="black", width=1)
    draw.text((720, y + 12), "Итого:", font=small, fill="black")
    draw.text((1000, y + 12), "18 450,00", font=small, fill="black")
    draw.text((100, y + 90), "Отпустил: кладовщик Смирнов А.В.", font=regular, fill="black")
    draw.text((100, y + 140), "Получил: экспедитор Кузнецов Д.Н.", font=regular, fill="black")
    draw.text(
        (100, y + 210),
        "Invoice / Packing list: PL-2025-0458",
        font=tiny,
        fill=(60, 60, 60),
    )
    save(img, "nakladnaya_tovarnaya.png")


if __name__ == "__main__":
    make_zayavlenie()
    make_spravka()
    make_nakladnaya()
    print("done")
