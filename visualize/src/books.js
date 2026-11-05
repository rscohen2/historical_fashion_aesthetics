import { useEffect, useState } from 'react'

// Book metadata shared by every page, `{ book_id: [title, author, year] }`,
// written by `python -m fashion.analysis.books`. Example sentences carry only
// their book id (`bk`) and are cited from here, so no page's data repeats it.
const BOOKS_URL = `${import.meta.env.BASE_URL}data/books.json`

// Fetched once per session, on first use.
let request = null

export function loadBooks() {
  if (!request) {
    request = fetch(BOOKS_URL).then((r) => {
      if (!r.ok) throw new Error(`${r.status} loading ${BOOKS_URL}`)
      return r.json()
    }).catch((err) => {
      request = null
      throw err
    })
  }
  return request
}

// The books map, or null while it loads. If it can't be loaded, an empty map,
// so citations fall back to book ids.
export function useBooks() {
  const [books, setBooks] = useState(null)
  useEffect(() => {
    let live = true
    loadBooks()
      .then((b) => live && setBooks(b))
      .catch((err) => {
        console.warn(`Book metadata unavailable (${err.message})`)
        if (live) setBooks({})
      })
    return () => { live = false }
  }, [])
  return books
}

// { title, author, year } for a book id; fields are null when unknown.
export function bookInfo(books, id) {
  const [title = null, author = null, year = null] = books?.[id] ?? []
  return { title, author, year }
}

// Examples are `{ s: sentence, bk: book_id, ... }`. Exports from before books.json
// existed stored bare sentence strings; treat those as examples with no book.
export const toExample = (ex) => (typeof ex === 'string' ? { s: ex } : ex)
