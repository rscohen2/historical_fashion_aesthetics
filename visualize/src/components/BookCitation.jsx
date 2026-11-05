import { bookInfo, useBooks } from '../books.js'
import './BookCitation.css'

// "Title — Author, 1885" for an example's book, from the shared books.json.
// Falls back to the book id when the book has no title.
export default function BookCitation({ id }) {
  const books = useBooks()
  if (!id) return null
  if (!books) return <cite className="book-cite">&nbsp;</cite>

  const { title, author, year } = bookInfo(books, id)
  const byline = [author, year].filter((v) => v != null).join(', ')
  const full = `${title ?? id}${byline ? ` — ${byline}` : ''}`
  return (
    <cite className="book-cite" title={title ? `${full} (${id})` : full}>
      {title ? <span className="book-cite-title">{title}</span> : id}
      {byline && ` — ${byline}`}
    </cite>
  )
}
