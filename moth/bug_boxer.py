#!/usr/bin/env python

import json
import tkinter as tk
from pathlib import Path
from tkinter import Event, filedialog, messagebox, ttk

import imagesize
from PIL import ImageTk

from moth.pylib import bbox

SCROLL_DOWN = 5  # Mouse event code for scrolling down


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.curr_dir: Path = Path()
        self.bbox_path: Path = None
        self.dirty = False
        self.dragging = False

        self.photo = None

        self.bbox_images: list[bbox.BBoxImage] = []
        self.image_no = tk.IntVar()

        self.bbox_content = tk.StringVar(value="moth")

        self.title("Create bounding boxes around insects on images.")

        self.canvas_frame = ttk.Frame(self, borderwidth=2)
        self.control_frame = ttk.Frame(self, borderwidth=2)

        self.canvas_frame.grid(column=0, row=0, sticky="nesw")
        self.control_frame.grid(column=1, row=0, sticky="nesw")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        self.h_bar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.v_bar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            background="black",
            yscrollcommand=self.v_bar.set,
            xscrollcommand=self.h_bar.set,
        )

        self.h_bar["command"] = self.canvas.xview
        self.v_bar["command"] = self.canvas.yview

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.h_bar.grid(row=1, column=0, sticky="ew")
        self.v_bar.grid(row=0, column=1, sticky="ns")

        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(1, weight=0)
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(1, weight=0)

        self.canvas.bind("<ButtonPress-1>", self.on_box_start)
        self.canvas.bind("<B1-Motion>", self.on_box_draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_box_done)
        self.canvas.bind("<ButtonRelease-3>", self.on_box_delete)  # right-click
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

        self.dir_button = tk.Button(
            self.control_frame,
            text="Choose image directory",
            font=bbox.FONT,
            command=self.get_image_dir,
        )
        self.load_button = tk.Button(
            self.control_frame,
            text="Load JSON",
            font=bbox.FONT,
            command=self.load,
        )
        self.save_label = ttk.Label(
            self.control_frame, text="", font=bbox.FONT_SM, wraplength=300
        )
        self.save_button = tk.Button(
            self.control_frame,
            text="Save",
            font=bbox.FONT,
            command=self.save,
        )
        self.save_as_button = tk.Button(
            self.control_frame,
            text="Save As...",
            font=bbox.FONT,
            command=self.save_as,
        )
        self.image_label = ttk.Label(
            self.control_frame, text="", font=bbox.FONT_SM, wraplength=300
        )
        self.validate_cmd = (self.register(self.validate_spinner), "%P")
        self.spinner = ttk.Spinbox(
            self.control_frame,
            textvariable=self.image_no,
            wrap=True,
            font=bbox.FONT,
            justify="center",
            state="disabled",
            command=self.display_image,
            takefocus=False,
            validate="all",
            validatecommand=self.validate_cmd,
        )
        self.content_label = ttk.Label(
            self.control_frame, text="Bug type:", font=bbox.FONT
        )

        self.dir_button.grid(row=0, sticky="nsew", padx=16, pady=16)
        self.load_button.grid(row=1, sticky="nsew", padx=16, pady=16)
        self.save_label.grid(row=2, sticky="nsew", padx=16, pady=16)
        self.save_button.grid(row=3, sticky="nsew", padx=16, pady=16)
        self.save_as_button.grid(row=4, sticky="nsew", padx=16, pady=16)
        self.image_label.grid(row=5, sticky="nsew", padx=16, pady=16)
        self.spinner.grid(row=6, sticky="nsew", padx=16, pady=16)
        self.content_label.grid(row=7, sticky="ew", padx=16, pady=16)

        style = ttk.Style(self)
        for i, (content_value, opts) in enumerate(bbox.BBOX_FORMAT.items(), 8):
            name = f"{content_value}.TRadiobutton"
            style.configure(name, **opts)
            radio = ttk.Radiobutton(
                self.control_frame,
                text=content_value.replace("_", " "),
                value=content_value,
                variable=self.bbox_content,
                style=name,
            )
            radio.grid(row=i, sticky="w", padx=32, pady=8)

        self.bind("<Key>", self.on_key)

        self.protocol("WM_DELETE_WINDOW", self.safe_quit)
        self.focus()
        self.unbind_all("<<NextWindow>>")

    def validate_spinner(self, value: str) -> bool:
        if not value.isdigit():
            return False
        no = int(value)
        ok = 1 <= no <= len(self.bbox_images)
        if ok:
            self.image_no.set(no)
            self.display_image()
        return ok

    def next_image(self) -> None:
        if not self.bbox_images:
            return
        no = self.image_no.get()
        no = min(no + 1, len(self.bbox_images))
        self.image_no.set(no)

    def prev_image(self) -> None:
        if not self.bbox_images:
            return
        no = self.image_no.get()
        no = max(no - 1, 1)
        no = 0 if not self.bbox_images else no
        self.image_no.set(no)

    def get_image_rec(self) -> bbox.BBoxImage:
        return self.bbox_images[self.image_no.get() - 1]

    def on_key(self, event: Event) -> None:
        match event.keysym:
            case "m" | "M":
                self.bbox_content.set("moth")
            case "n" | "N":
                self.bbox_content.set("not_moth")
            case "u" | "U":
                self.bbox_content.set("unsure")
            case "Left" | "Down" | "a" | "A" | "h" | "H":
                self.prev_image()
                self.display_image()
            case "Right" | "Up" | "g" | "G" | "l" | "L":
                self.next_image()
                self.display_image()
            case "q" | "Q" | "Escape":
                self.safe_quit()

    def on_mouse_wheel(self, event: Event) -> None:
        width = self.canvas.winfo_reqwidth()
        height = self.canvas.winfo_reqheight()
        direction = 1 if event.num == SCROLL_DOWN else -1
        if int(event.state) & 0x01:
            self.canvas.xview_scroll(direction * int(width / 100), "units")
        else:
            self.canvas.yview_scroll(direction * int(height / 100), "units")

    def get_image_dir(self) -> None:
        image_dir = filedialog.askdirectory(
            initialdir=self.curr_dir,
            title="Choose image directory",
        )
        if not image_dir:
            return

        self.curr_dir = Path(image_dir)
        self.dirty = False

        self.bbox_images = []
        for path in sorted(self.curr_dir.glob("*")):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                width, height = imagesize.get(path)
                self.bbox_images.append(
                    bbox.BBoxImage(path=str(path), width=int(width), height=int(height))
                )

        if self.bbox_images:
            self.spinner_setup()
            self.save_button.configure(state="normal")
            self.save_as_button.configure(state="normal")
            self.display_image()
        else:
            self.spinner_clear()
            self.save_button.configure(state="disabled")
            self.save_as_button.configure(state="disabled")
            self.canvas.delete("all")
            self.image_label.configure(text="")

    def display_image(self) -> None:
        bbox_image: bbox.BBoxImage = self.get_image_rec()
        self.image_label.configure(text=Path(bbox_image.path).name)
        self.photo = ImageTk.PhotoImage(file=bbox_image.path)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas["scrollregion"] = (0, 0, bbox_image.width, bbox_image.height)
        self.display_image_boxes(bbox_image)

    def clamp(self, event: Event, bbox_image: bbox.BBoxImage) -> tuple[int, int]:
        x = max(self.canvas.canvasx(event.x), 0)
        x = min(x, bbox_image.width - 1)
        y = max(self.canvas.canvasy(event.y), 0)
        y = min(y, bbox_image.height - 1)
        return x, y

    def on_box_start(self, event: Event) -> None:
        if not self.bbox_images:
            return

        bbox_image: bbox.BBoxImage = self.get_image_rec()
        x, y = self.clamp(event, bbox_image)

        bbox_content = self.bbox_content.get()
        color = bbox.BBOX_COLOR[bbox_content]
        id_ = self.canvas.create_rectangle(
            0, 0, 1, 1, outline=color, width=4, tags=("box", bbox_content)
        )
        bbox_image.bboxes.append(
            bbox.BBox(id_=id_, x0=x, y0=y, x1=x, y1=y, content=bbox_content)
        )

        self.dirty = True
        self.dragging = True

    def on_box_draw(self, event: Event) -> None:
        if self.dragging and self.bbox_images:
            bbox_image: bbox.BBoxImage = self.get_image_rec()
            if bbox_image.bboxes:
                box = bbox_image.bboxes[-1]
                box.x1, box.y1 = self.clamp(event, bbox_image)
                self.canvas.coords(box.id_, box.x0, box.y0, box.x1, box.y1)

    def on_box_done(self, _: Event) -> None:
        if self.dragging and self.bbox_images:
            bbox_image: bbox.BBoxImage = self.get_image_rec()
            bbox_image.filter_size()
            self.dragging = False
            self.display_image_boxes(bbox_image)

    def on_box_delete(self, event: Event) -> None:
        bbox_image: bbox.BBoxImage = self.get_image_rec()

        x = self.canvas.canvasx(event.x)  # Not clamped so clicks out of bounds...
        y = self.canvas.canvasy(event.y)  # do not delete a bbox

        self.dirty = True
        bbox_image.delete_box(x, y)
        self.display_image_boxes(bbox_image)

    def display_image_boxes(self, bbox_image: bbox.BBoxImage) -> None:
        self.canvas.delete("box")
        for box in bbox_image.bboxes:
            self.canvas.create_rectangle(
                box.x0,
                box.y0,
                box.x1,
                box.y1,
                width=4,
                outline=bbox.BBOX_COLOR[box.content],
                tags=("box", box.content),
            )

    def save(self) -> None:
        if not self.bbox_path:
            return

        self.curr_dir = self.bbox_path.parent
        self.dirty = False

        output = [i.to_dict() for i in self.bbox_images]

        with self.bbox_path.open("w") as out_json:
            json.dump(output, out_json, indent=4)

    def save_as(self) -> None:
        if not self.bbox_images:
            return

        bbox_path = filedialog.asksaveasfilename(
            initialdir=self.curr_dir,
            title="Save image boxes as...",
            filetypes=(("json", "*.json"), ("all files", "*")),
        )

        if not bbox_path:
            return

        self.bbox_path = Path(bbox_path)
        self.save_label.configure(text=self.bbox_path.name)
        self.save()

    def load(self) -> None:
        bbox_path = filedialog.askopenfilename(
            initialdir=self.curr_dir,
            title="Load image boxes",
            filetypes=(("json", "*.json"), ("all files", "*")),
        )
        if not bbox_path:
            return

        self.bbox_path = Path(bbox_path)
        self.curr_dir = self.bbox_path.parent

        with self.bbox_path.open() as in_json:
            json_images = json.load(in_json)

        self.dirty = False
        self.bbox_images = []

        try:
            self.bbox_images = [bbox.BBoxImage.load_json(i) for i in json_images]
            for data in json_images:
                bbox_image = bbox.BBoxImage.load_json(data)
                self.bbox_images.append(bbox_image)

            self.spinner_setup()
            self.save_button.configure(state="normal")
            self.save_as_button.configure(state="normal")
            self.display_image()

        except KeyError:
            messagebox.showerror(
                title="Load error", message="Could not load the JSON file\n\n"
            )
            self.bbox_images = []
            self.save_button.configure(state="disabled")
            self.save_as_button.configure(state="disabled")
            self.spinner_clear()
            self.canvas.delete("all")

    def spinner_setup(self) -> None:
        self.image_no.set(1)
        self.spinner.configure(state="normal")
        self.spinner.configure(from_=1)
        self.spinner.configure(to=len(self.bbox_images))

    def spinner_clear(self) -> None:
        self.image_no.set(0)
        self.spinner.configure(state="disabled")
        self.spinner.configure(from_=0)
        self.spinner.configure(to=0)

    def safe_quit(self) -> None:
        if self.dirty:
            yes = messagebox.askyesno(
                self.title(), "Are you sure you want to exit without saving?\n\n"
            )
            if not yes:
                return
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
