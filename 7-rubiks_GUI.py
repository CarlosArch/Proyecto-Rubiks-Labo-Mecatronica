import tkinter as tk

root = tk.Tk(baseName='Rubiks')

header = tk.Frame(root)
header.pack()
author = tk.Label(header, text='Autor: Carlos Daniel Archundia Cejudo')
author.pack()

content = tk.Frame(root)
content.pack()

rows = []
buttons = []
i = 0
for r in range(3):
    rows.append(tk.Frame(content))
    rows[r].pack()
    for c in range(3):
        pixel = tk.PhotoImage(width=1, height=1)
        buttons.append(
            tk.Button(
                rows[r],
                bg='red',
                image=pixel,
                width=100,
                height=100
                )
            )
        buttons[i].pack(side='left')
        i += 1



root.mainloop()

