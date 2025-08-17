export interface Video {
  id: number;
  title: string;
  thumbnailUrl: string;
  tags: string[];
  description: string;
}

export const dummyVideos: Video[] = [
  {
    id: 1,
    title: "【VR】タイトル1",
    thumbnailUrl: "/placeholder.svg",
    tags: ["#素人", "#制服", "#VR対応"],
    description: "これは動画の説明文です。これは動画の説明文です。これは動画の説明文です。",
  },
  {
    id: 2,
    title: "タイトル2",
    thumbnailUrl: "/placeholder.svg",
    tags: ["#人妻", "#巨乳"],
    description: "これは動画の説明文です。これは動画の説明文です。",
  },
  {
    id: 3,
    title: "タイトル3",
    thumbnailUrl: "/placeholder.svg",
    tags: ["#JK", "#ギャル"],
    description: "これは動画の説明文です。",
  },
  {
    id: 4,
    title: "タイトル4",
    thumbnailUrl: "/placeholder.svg",
    tags: ["#アニメ", "#コスプレ"],
    description: "これは動画の説明文です。これは動画の説明文です。これは動画の説明文です。これは動画の説明文です。",
  },
  {
    id: 5,
    title: "タイトル5",
    thumbnailUrl: "/placeholder.svg",
    tags: ["#熟女", "#温泉"],
    description: "これは動画の説明文です。これは動画の説明文です。",
  },
];
